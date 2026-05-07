//! Scheduler hot-path bench.
//!
//! Measures `Scheduler::compute_schedule` + `apply_schedule` over realistic
//! batch sizes. The scheduler is called once per engine step, so its
//! per-call cost directly bounds achievable steps/sec — and therefore
//! decode tps × concurrency.
//!
//! What this measures:
//!
//! - `compute_schedule_*` — pure read-only schedule decision over a populated
//!   waiting queue + running set; representative of the warm-state workload
//!   inside `engine::strategy::run_engine_loop`.
//! - `add_request_then_schedule` — full add/schedule cycle for the
//!   request-arrival hot path.
//!
//! Pure CPU; no GPU dependency. Numbers are relative.

use std::collections::HashMap;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use vllm_core::request::{RequestId, RequestStatus, SequenceState};
use vllm_core::scheduler::{Scheduler, SchedulerConfig};

const BLOCK_SIZE: usize = 16;
const PROMPT_LEN: usize = 256;
const FREE_BLOCKS: usize = 4096;

fn make_state(id: RequestId, arrival_order: u64) -> SequenceState {
    let prompt = vec![1u32; PROMPT_LEN];
    let mut s = SequenceState::new(id, prompt, 256, 2, BLOCK_SIZE, arrival_order);
    s.status = RequestStatus::Decoding;
    s
}

fn populate(scheduler: &mut Scheduler, batch: usize) -> Vec<SequenceState> {
    let mut owned: Vec<SequenceState> = Vec::with_capacity(batch);
    for i in 0..batch {
        let id = i as RequestId;
        scheduler.add_request(id);
        owned.push(make_state(id, i as u64));
    }
    owned
}

fn states_map(owned: &[SequenceState]) -> HashMap<RequestId, &SequenceState> {
    owned.iter().map(|s| (s.request_id, s)).collect()
}

fn bench_compute_schedule(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_schedule");
    for &batch in &[1usize, 4, 8, 16, 32, 64] {
        let cfg = SchedulerConfig {
            max_running_requests: 256,
            max_tokens_per_step: 8192,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(cfg);
        let owned = populate(&mut scheduler, batch);
        // Promote everyone to running so compute_schedule sees a warm decode set.
        let decision = scheduler.compute_schedule(&states_map(&owned), FREE_BLOCKS);
        scheduler.apply_schedule(&decision);

        let states = states_map(&owned);
        group.bench_with_input(BenchmarkId::new("batch", batch), &batch, |b, _| {
            b.iter(|| {
                let d = scheduler.compute_schedule(black_box(&states), FREE_BLOCKS);
                black_box(d);
            });
        });
    }
    group.finish();
}

fn bench_add_then_schedule(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_then_schedule");
    for &batch in &[1usize, 4, 16, 64] {
        group.bench_with_input(BenchmarkId::new("batch", batch), &batch, |b, _| {
            b.iter_with_setup(
                || {
                    let scheduler = Scheduler::new(SchedulerConfig::default());
                    let owned: Vec<SequenceState> = (0..batch)
                        .map(|i| make_state(i as RequestId, i as u64))
                        .collect();
                    (scheduler, owned)
                },
                |(mut scheduler, owned)| {
                    for s in &owned {
                        scheduler.add_request(s.request_id);
                    }
                    let states = states_map(&owned);
                    let d = scheduler.compute_schedule(&states, FREE_BLOCKS);
                    scheduler.apply_schedule(&d);
                    black_box(scheduler);
                },
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_compute_schedule, bench_add_then_schedule);
criterion_main!(benches);
