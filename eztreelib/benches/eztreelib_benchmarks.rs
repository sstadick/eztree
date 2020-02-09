#[macro_use]
extern crate criterion;
extern crate eztreelib;
extern crate rand;

use criterion::Criterion;
use eztreelib::EzTree;
use ginterval::Interval;
use rand::Rng;

type Iv = Interval<bool>;

fn randomi(imin: u32, imax: u32) -> u32 {
    let mut rng = rand::thread_rng();
    imin + rng.gen_range(0, imax - imin)
}

fn make_random(n: usize, range_max: u32, size_min: u32, size_max: u32) -> Vec<Iv> {
    let mut result = Vec::with_capacity(n);
    for _i in 0..n {
        let s = randomi(0, range_max);
        let e = s + randomi(size_min, size_max);
        result.push(Interval {
            start: s,
            stop: e,
            val: false,
        });
    }
    result
}

fn make_interval_set() -> (Vec<Iv>, Vec<Iv>) {
    //let n = 3_000_000;
    let n = 50_000;
    let chrom_size = 100_000_000;
    let min_interval_size = 500;
    let max_interval_size = 80000;
    let intervals = make_random(n, chrom_size, min_interval_size, max_interval_size);
    let other_intervals = make_random(n, 10 * chrom_size, 1, 2);
    (intervals, other_intervals)
}

pub fn query(c: &mut Criterion) {
    let s_size = 10;
    let (intervals, other_intervals) = make_interval_set();
    // Make Lapper intervals
    let mut bad_intervals: Vec<Iv> = intervals
        .iter()
        .map(|iv| Interval {
            start: iv.start,
            stop: iv.stop,
            val: true,
        })
        .collect();
    bad_intervals.push(Iv {
        start: 0,
        stop: 90_000_000,
        val: false,
    });

    let eztree = EzTree::new(intervals.iter().map(|iv| Interval { ..*iv }).collect());
    let other_eztree = EzTree::new(
        other_intervals
            .iter()
            .map(|iv| Interval { ..*iv })
            .collect(),
    );
    let bad_eztree = EzTree::new(bad_intervals.iter().map(|iv| Interval { ..*iv }).collect());

    let mut comparison_group = c.benchmark_group("Bakeoff");
    comparison_group
        .sample_size(s_size)
        .bench_function("EzTree: find with 100% hit rate", |b| {
            b.iter(|| {
                for x in intervals.iter() {
                    eztree.find(x.start, x.stop).len();
                }
            });
        });
    comparison_group.sample_size(s_size).bench_function(
        "EzTree: find with below 100% hit rate",
        |b| {
            b.iter(|| {
                for x in other_intervals.iter() {
                    eztree.find(x.start, x.stop).len();
                }
            });
        },
    );
    comparison_group.sample_size(s_size).bench_function(
        "EzTree: find with below 100% hit rate - chromosome spanning interval",
        |b| {
            b.iter(|| {
                for x in other_intervals.iter() {
                    bad_eztree.find(x.start, x.stop).len();
                }
            });
        },
    );
    comparison_group.finish();
}
criterion_group!(benches, query);
criterion_main!(benches);
