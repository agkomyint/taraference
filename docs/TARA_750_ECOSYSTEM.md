# Tara ↔ taraference 750 tok/s ecosystem

Training and co-design live in Tara HQ:

```text
D:\Tara_HQ\departments\taraference_750_department\
```

## Why

3B/4B models on this stack top out well below 750 single-stream tok/s (physics + measured).  
The 750 vehicle is a **~70–120M active-param** Tara-Sprint dense (or MoE with same active budget).

## Engine follow-ups

See `taraference_750_department/taraference/CODESIGN.md`:

1. **P0:** Llama-family dense GGUF path (Tara-Sprint-80 export)  
2. **P1:** Keep quant-friendly dims (already in configs)  
3. **P2:** MoE sparse expert runtime for 300–500M *total* / ~120M *active*  

## Metric

Always `single_stream_accepted_tps` per `GOAL.md`. Never confuse with aggregate multi-user throughput.
