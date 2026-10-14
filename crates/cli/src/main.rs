//! `tarafer`: lean local runtime for small specialised AI models.
//!
//! - `cmd` command layer, `bench` speed report, `serve` OpenAI-compatible server
//! - `download` model registry, `tts_commands` / `image_commands` task front-ends

mod bench;
mod cmd;
mod download;
mod image_commands;
mod self_update;
mod serve;
mod tts_commands;

fn main() {
    if let Err(error) = cmd::run() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}
