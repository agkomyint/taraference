// Portable fp32 compute, original GGUF packed weights, packed fp16 KV.
// Logical groups of 32 lanes, reduced with workgroup barriers (not subgroups).
override WEIGHT_KIND: u32 = 255u;
override TILE_K: u32 = 16u;
struct Params {
    n: u32, m: u32, kind: u32, offset: u32,
    eps: f32, theta: f32, heads: u32, kvheads: u32,
    dim: u32, maxseq: u32, bias: u32, unused0: u32,
    unused1: u32, unused2: u32, unused3: u32, unused4: u32,
};
struct State { pos: u32, token: u32, count: u32, unused1: u32 };
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@group(0) @binding(4) var<storage, read_write> kv: array<u32>;
@group(0) @binding(5) var<uniform> p: Params;
@group(0) @binding(6) var<uniform> state: State;
@group(0) @binding(7) var<storage, read> up_weights: array<u32>;

fn byte_at(i: u32) -> u32 { return (w[i / 4u] >> ((i % 4u) * 8u)) & 255u; }
fn signed_byte(i: u32) -> f32 { let x = byte_at(i); return f32(i32(x) - select(0, 256, x >= 128u)); }
fn half_at(i: u32) -> f32 { return unpack2x16float(byte_at(i) | (byte_at(i + 1u) << 8u)).x; }
fn scale_min(base: u32, j: u32) -> vec2<f32> {
    if j < 4u { return vec2<f32>(f32(byte_at(base + j) & 63u), f32(byte_at(base + j + 4u) & 63u)); }
    return vec2<f32>(f32((byte_at(base + j + 4u) & 15u) | ((byte_at(base + j - 4u) >> 6u) << 4u)),
                     f32((byte_at(base + j + 4u) >> 4u) | ((byte_at(base + j) >> 6u) << 4u)));
}
fn weight(index: u32) -> f32 {
    let kind = select(WEIGHT_KIND, p.kind, WEIGHT_KIND == 255u);
    if kind == 0u { return bitcast<f32>(w[index]); }
    if kind == 1u { return half_at(index * 2u); }
    if kind == 8u { let base = (index / 32u) * 34u; return half_at(base) * signed_byte(base + 2u + index % 32u); }
    if kind == 2u {
        let base = (index / 32u) * 18u; let j = index % 32u;
        let q = (byte_at(base + 2u + j % 16u) >> ((j / 16u) * 4u)) & 15u;
        return half_at(base) * f32(i32(q) - 8);
    }
    if kind == 12u || kind == 13u {
        let block_bytes = select(144u, 176u, kind == 13u);
        let base = (index / 256u) * block_bytes; let j = index % 256u;
        let group = j / 32u; let lane = j % 32u;
        let qs = base + select(16u, 48u, kind == 13u);
        var q = (byte_at(qs + (j / 64u) * 32u + lane) >> ((group % 2u) * 4u)) & 15u;
        if kind == 13u { q += ((byte_at(base + 16u + lane) >> group) & 1u) * 16u; }
        let sm = scale_min(base + 4u, group);
        return half_at(base) * sm.x * f32(q) - half_at(base + 2u) * sm.y;
    }
    // Q6_K: [128 low bytes,64 high bytes,16 signed scales,fp16 d].
    let base = (index / 256u) * select(210u,256u,kind==114u); let j = index % 256u;
    let half_block = j / 128u; let part = (j % 128u) / 32u; let lane = j % 32u;
    let low = byte_at(base + half_block * 64u + (part % 2u) * 32u + lane);
    let high = byte_at(base + 128u + half_block * 32u + lane);
    let q = ((low >> ((part / 2u) * 4u)) & 15u) | (((high >> (part * 2u)) & 3u) << 4u);
    let scale_index=half_block*8u+part*2u+lane/16u;
    if kind==114u { return bitcast<f32>(w[base/4u+48u+scale_index])*f32(i32(q)-32); }
    let sc = signed_byte(base + 192u + scale_index);
    return half_at(base + 208u) * sc * f32(i32(q) - 32);
}

var<workgroup> red: array<f32, 256>;
var<workgroup> ids: array<u32, 256>;
var<workgroup> scores: array<f32, 2048>;

@compute @workgroup_size(128)
fn embed(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x < p.n {
        var token = state.token;
        if state.count > 1u { token = u32(b[gid.y]); }
        o[gid.y * p.n + gid.x] = weight(token * p.n + gid.x);
    }
}

// Four output rows per workgroup, 32 lanes per row. No NVIDIA warp assumptions.
@compute @workgroup_size(128)
fn gemv(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
    let lane = lid.x % 32u; let row = group.x * 4u + lid.x / 32u;
    var sum = 0.0;
    if row < p.m {
        for (var col = lane; col < p.n; col += 32u) { sum += weight(row * p.n + col) * a[col]; }
    }
    red[lid.x] = sum; workgroupBarrier();
    for (var s = 16u; s > 0u; s /= 2u) {
        if lane < s { red[lid.x] += red[lid.x + s]; }
        workgroupBarrier();
    }
    if lane == 0u && row < p.m { var result = red[lid.x]; if p.bias != 0u { result += b[row]; } if p.unused0 != 0u { result += o[row]; } o[row] = result; }
}

// Block-specialized GEMV: unpack each scale once per 256-value block and
// each quant byte once per pair/four values, rather than scalar weight() calls.
fn q4_pair(base: u32, group: u32, col: u32, lane: u32, d: vec2<f32>) -> f32 {
    let sm0 = scale_min(base + 4u, group * 2u);
    let sm1 = scale_min(base + 4u, group * 2u + 1u);
    let bits = byte_at(base + 16u + group * 32u + lane);
    return (d.x * sm0.x * f32(bits & 15u) - d.y * sm0.y) * a[col + group * 64u + lane]
         + (d.x * sm1.x * f32(bits >> 4u) - d.y * sm1.y) * a[col + group * 64u + lane + 32u];
}
fn up_byte(i: u32) -> u32 { return (up_weights[i/4u] >> ((i%4u)*8u)) & 255u; }
fn up_scale(base: u32, j: u32) -> vec2<f32> {
    if j < 4u { return vec2<f32>(f32(up_byte(base+j)&63u), f32(up_byte(base+j+4u)&63u)); }
    return vec2<f32>(f32((up_byte(base+j+4u)&15u)|((up_byte(base+j-4u)>>6u)<<4u)),
                     f32((up_byte(base+j+4u)>>4u)|((up_byte(base+j)>>6u)<<4u)));
}
fn q4_gate_up_pair(base: u32, group: u32, col: u32, lane: u32, dg: vec2<f32>, du: vec2<f32>) -> vec2<f32> {
    let g0=scale_min(base+4u,group*2u); let g1=scale_min(base+4u,group*2u+1u);
    let u0=up_scale(base+4u,group*2u); let u1=up_scale(base+4u,group*2u+1u);
    let g=byte_at(base+16u+group*32u+lane); let u=up_byte(base+16u+group*32u+lane);
    let x0=a[col+group*64u+lane]; let x1=a[col+group*64u+lane+32u];
    return vec2<f32>((dg.x*g0.x*f32(g&15u)-dg.y*g0.y)*x0+(dg.x*g1.x*f32(g>>4u)-dg.y*g1.y)*x1,
                     (du.x*u0.x*f32(u&15u)-du.y*u0.y)*x0+(du.x*u1.x*f32(u>>4u)-du.y*u1.y)*x1);
}
var<workgroup> up_red: array<f32,128>;
// The paired projections share input loads, expose independent accumulation
// chains, and write SwiGLU directly. Original packed weights remain separate.
@compute @workgroup_size(128)
fn gate_up_q4k(@builtin(local_invocation_id) lid:vec3<u32>,@builtin(workgroup_id) group:vec3<u32>) {
    let lane=lid.x%32u; let row=group.x*4u+lid.x/32u;
    var sum=vec2<f32>(0.0);
    if row<p.m {
        for(var block=0u;block<p.n/256u;block++) {
            let base=(row*(p.n/256u)+block)*144u; let col=block*256u;
            let dg=unpack2x16float(w[base/4u]); let du=unpack2x16float(up_weights[base/4u]);
            sum+=q4_gate_up_pair(base,0u,col,lane,dg,du);
            sum+=q4_gate_up_pair(base,1u,col,lane,dg,du);
            sum+=q4_gate_up_pair(base,2u,col,lane,dg,du);
            sum+=q4_gate_up_pair(base,3u,col,lane,dg,du);
        }
    }
    red[lid.x]=sum.x; up_red[lid.x]=sum.y; workgroupBarrier();
    for(var s=16u;s>0u;s/=2u) {
        if lane<s { red[lid.x]+=red[lid.x+s]; up_red[lid.x]+=up_red[lid.x+s]; }
        workgroupBarrier();
    }
    if lane==0u && row<p.m { let g=red[lid.x]; o[row]=(g/(1.0+exp(-g)))*up_red[lid.x]; }
}
fn reduce_gemv(sum: f32, row: u32, lid: u32) {
    let lanes = 32u;
    let lane = lid % lanes;
    red[lid] = sum; workgroupBarrier();
    for (var s = lanes / 2u; s > 0u; s /= 2u) { if lane < s { red[lid] += red[lid + s]; } workgroupBarrier(); }
    if lane == 0u && row < p.m {
        var result = red[lid];
        if p.bias != 0u { result += b[row]; }
        if p.unused0 != 0u { result += o[row]; }
        o[row] = result;
    }
}
@compute @workgroup_size(128)
fn gemv_q4k(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
    let lane = lid.x % 32u; let row = group.x * 4u + lid.x / 32u;
    var sum = 0.0;
    if row < p.m {
        for (var block = 0u; block < p.n / 256u; block++) {
            let base = (row * (p.n / 256u) + block) * 144u;
            let d = unpack2x16float(w[base / 4u]); let col = block * 256u;
            sum += q4_pair(base, 0u, col, lane, d);
            sum += q4_pair(base, 1u, col, lane, d);
            sum += q4_pair(base, 2u, col, lane, d);
            sum += q4_pair(base, 3u, col, lane, d);
        }
    }
    reduce_gemv(sum, row, lid.x);
}
@compute @workgroup_size(128)
fn gemv_q6k(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
    let lane = lid.x % 32u; let row = group.x * 4u + lid.x / 32u;
    var sum = 0.0;
    if row < p.m {
        for (var block = 0u; block < p.n / 256u; block++) {
            let base = (row * (p.n / 256u) + block) * 210u;
            let packed_d = unpack2x16float(w[(base + 208u) / 4u]);
            let d = select(packed_d.x, packed_d.y, (base + 208u) % 4u == 2u);
            for (var half_block = 0u; half_block < 2u; half_block++) {
                let lo0 = byte_at(base + half_block * 64u + lane);
                let lo1 = byte_at(base + half_block * 64u + 32u + lane);
                let high = byte_at(base + 128u + half_block * 32u + lane);
                let scales = base + 192u + half_block * 8u + lane / 16u;
                let col = block * 256u + half_block * 128u + lane;
                sum += d * signed_byte(scales) * f32(i32((lo0 & 15u) | ((high & 3u) << 4u)) - 32) * a[col];
                sum += d * signed_byte(scales + 2u) * f32(i32((lo1 & 15u) | (((high >> 2u) & 3u) << 4u)) - 32) * a[col + 32u];
                sum += d * signed_byte(scales + 4u) * f32(i32((lo0 >> 4u) | (((high >> 4u) & 3u) << 4u)) - 32) * a[col + 64u];
                sum += d * signed_byte(scales + 6u) * f32(i32((lo1 >> 4u) | (((high >> 6u) & 3u) << 4u)) - 32) * a[col + 96u];
            }
        }
    }
    reduce_gemv(sum, row, lid.x);
}

// Lossless Q6_K device layout: 192 original payload bytes followed by 16
// precomputed fp32 (fp16 d * signed scale) coefficients. All blocks align to 256 B.
@compute @workgroup_size(128)
fn gemv_q6k_scaled(@builtin(local_invocation_id) lid:vec3<u32>,@builtin(workgroup_id) group:vec3<u32>) {
    let lane=lid.x%32u; let row=group.x*4u+lid.x/32u; var sum=0.0;
    if row<p.m {
        for(var block=0u;block<p.n/256u;block++) {
            let base=(row*(p.n/256u)+block)*256u;
            for(var half_block=0u;half_block<2u;half_block++) {
                let lo0=byte_at(base+half_block*64u+lane);
                let lo1=byte_at(base+half_block*64u+32u+lane);
                let hi=byte_at(base+128u+half_block*32u+lane);
                let sc=base/4u+48u+half_block*8u+lane/16u;
                let col=block*256u+half_block*128u+lane;
                sum+=bitcast<f32>(w[sc])*f32(i32((lo0&15u)|((hi&3u)<<4u))-32)*a[col];
                sum+=bitcast<f32>(w[sc+2u])*f32(i32((lo1&15u)|(((hi>>2u)&3u)<<4u))-32)*a[col+32u];
                sum+=bitcast<f32>(w[sc+4u])*f32(i32((lo0>>4u)|(((hi>>4u)&3u)<<4u))-32)*a[col+64u];
                sum+=bitcast<f32>(w[sc+6u])*f32(i32((lo1>>4u)|(((hi>>6u)&3u)<<4u))-32)*a[col+96u];
            }
        }
    }
    reduce_gemv(sum,row,lid.x);
}

var<workgroup> tile_x: array<f32, 1040>;
// Padding removes stride-16 bank conflicts when threads read different rows.
var<workgroup> tile_w: array<f32, 1040>;
// Tiled [tokens,K] x [rows,K]^T. Decode uses the block GEMV above.
// Each quantized weight tile is reused for up to 16 prompt tokens.
@compute @workgroup_size(16, 16)
fn gemm(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
    let row = group.x * 16u + lid.x; let token = group.y * 16u + lid.y;
    var sum = 0.0;
    for (var k0 = 0u; k0 < p.n; k0 += TILE_K) {
        for (var slice = 0u; slice < TILE_K / 16u; slice++) {
            let col = lid.x + slice * 16u;
            var x = 0.0; var wt = 0.0;
            if token < state.count && k0 + col < p.n { x = a[token * p.n + k0 + col]; }
            let weight_row = group.x * 16u + lid.y;
            if weight_row < p.m && k0 + col < p.n { wt = weight(weight_row * p.n + k0 + col); }
            tile_x[lid.y * (TILE_K + 1u) + col] = x;
            tile_w[lid.y * (TILE_K + 1u) + col] = wt;
        }
        workgroupBarrier();
        for (var j = 0u; j < TILE_K; j++) { sum += tile_w[lid.x * (TILE_K + 1u) + j] * tile_x[lid.y * (TILE_K + 1u) + j]; }
        workgroupBarrier();
    }

    if row < p.m && token < state.count {
        if p.bias != 0u { sum += b[row]; }
        if p.unused0 != 0u { sum += o[token * p.m + row]; }
        o[token * p.m + row] = sum;
    }
}

@compute @workgroup_size(256)
fn rms(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
    let token = group.x;
    let input_token = token + select(0u, max(state.count, 1u) - 1u, p.offset != 0u);
    var sum = 0.0;
    for (var i = lid.x; i < p.n; i += 256u) { sum += a[input_token * p.n + i] * a[input_token * p.n + i]; }
    red[lid.x] = sum; workgroupBarrier();
    for (var s = 128u; s > 0u; s /= 2u) { if lid.x < s { red[lid.x] += red[lid.x + s]; } workgroupBarrier(); }
    let scale = inverseSqrt(red[0] / f32(p.n) + p.eps);
    for (var i = lid.x; i < p.n; i += 256u) { o[token * p.n + i] = a[input_token * p.n + i] * scale * bitcast<f32>(w[i]); }
}

@compute @workgroup_size(128)
fn rope(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_dim = p.dim / 2u;
    if gid.x >= p.heads * half_dim { return; }
    let j = gid.x % half_dim; let start = gid.y * p.heads * p.dim + (gid.x / half_dim) * p.dim + j;
    let angle = f32(state.pos + gid.y) * pow(p.theta, -2.0 * f32(j) / f32(p.dim));
    let c = cos(angle); let s = sin(angle);
    let x = o[start]; let y = o[start + half_dim];
    o[start] = x * c - y * s; o[start + half_dim] = x * s + y * c;
}

@compute @workgroup_size(128)
fn store_kv(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= p.n / 2u { return; }
    let start = (state.pos + gid.y) * (p.n / 2u) + gid.x;
    let i = gid.y * p.n + 2u * gid.x;
    kv[start] = pack2x16float(vec2<f32>(a[i], a[i + 1u]));
    kv[p.maxseq * (p.n / 2u) + start] = pack2x16float(vec2<f32>(b[i], b[i + 1u]));
}
fn cached(t: u32, head: u32, dim: u32, value: bool) -> f32 {
    let stride = p.kvheads * p.dim;
    let i = t * stride + head * p.dim + dim;
    let base = select(0u, p.maxseq * stride / 2u, value);
    let pair = unpack2x16float(kv[base + i / 2u]);
    return select(pair.x, pair.y, i % 2u != 0u);
}

@compute @workgroup_size(128)
fn attention(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
    let head = group.x; let kvhead = head / (p.heads / p.kvheads);
    let pos = state.pos + group.y;
    let query_base = group.y * p.heads * p.dim + head * p.dim;
    var max_score = -3.402823e38;
    for (var t = lid.x; t <= pos; t += 128u) {
        var dot = 0.0;
        for (var j = 0u; j < p.dim; j++) { dot += a[query_base + j] * cached(t, kvhead, j, false); }
        dot *= inverseSqrt(f32(p.dim)); scores[t] = dot; max_score = max(max_score, dot);
    }
    red[lid.x] = max_score; workgroupBarrier();
    for (var s = 64u; s > 0u; s /= 2u) { if lid.x < s { red[lid.x] = max(red[lid.x], red[lid.x + s]); } workgroupBarrier(); }
    let maximum = red[0]; var sum = 0.0;
    for (var t = lid.x; t <= pos; t += 128u) { let v = exp(scores[t] - maximum); scores[t] = v; sum += v; }
    workgroupBarrier(); red[lid.x] = sum; workgroupBarrier();
    for (var s = 64u; s > 0u; s /= 2u) { if lid.x < s { red[lid.x] += red[lid.x + s]; } workgroupBarrier(); }
    let denom = red[0];
    for (var j = lid.x; j < p.dim; j += 128u) {
        var value = 0.0;
        for (var t = 0u; t <= pos; t++) { value += scores[t] * cached(t, kvhead, j, true); }
        o[query_base + j] = value / denom;
    }
}

@compute @workgroup_size(128)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) { if gid.x < p.n { let i = gid.y * p.n + gid.x; o[i] += a[i]; } }
@compute @workgroup_size(128)
fn swiglu(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x < p.n { let i = gid.y * p.n + gid.x; let x = a[i]; o[i] = (x / (1.0 + exp(-x))) * b[i]; }
}

@compute @workgroup_size(256)
fn argmax(@builtin(local_invocation_id) lid: vec3<u32>) {
    var best = -3.402823e38; var id = 0u;
    for (var i = lid.x; i < p.n; i += 256u) { if a[i] > best || (a[i] == best && i < id) { best = a[i]; id = i; } }
    red[lid.x] = best; ids[lid.x] = id; workgroupBarrier();
    for (var s = 128u; s > 0u; s /= 2u) {
        if lid.x < s {
            if red[lid.x + s] > red[lid.x] || (red[lid.x + s] == red[lid.x] && ids[lid.x + s] < ids[lid.x]) {
                red[lid.x] = red[lid.x + s]; ids[lid.x] = ids[lid.x + s];
            }
        }
        workgroupBarrier();
    }
    if lid.x == 0u { o[0] = bitcast<f32>(ids[0]); }
}

// GPU fixture entry point: exercise every packing layout independently of GEMV.
@compute @workgroup_size(128)
fn unpack(@builtin(global_invocation_id) gid: vec3<u32>) { if gid.x < p.n { o[gid.x] = weight(gid.x); } }
