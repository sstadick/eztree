use bio::io::bed;
use eztreelib::EzTree;
use fnv::FnvHashMap;
use ginterval::Interval;
use std::process;

type Tree<T> = FnvHashMap<String, EzTree<T>>;

fn make_tree(input: &str) -> EzTree<usize> {
    let mut reader = bed::Reader::from_file(input).expect("Couldn't open the input file");
    let mut records = vec![];
    let mut intervals = vec![];
    let mut counter = 0;
    for record in reader.records() {
        let rec = record.unwrap();
        let interval = Interval {
            start: rec.start() as u32,
            stop: rec.end() as u32,
            val: counter,
        };
        counter += 1;
        intervals.push(interval);
        records.push(rec);
    }

    let tree = EzTree::new(intervals);
    tree
}

// Simple main function for intersecting two bed files
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Must input bed files.");
        process::exit(1);
    }
    let bed_1 = &args[1];
    let tree = make_tree(&bed_1);

    // Do the compare
    let bed_2 = &args[2];
    let mut reader = bed::Reader::from_file(bed_2).expect("Couldn't open the input file");
    for record in reader.records() {
        let rec = record.ok().expect("Error reading record.");
        let result = tree.find(rec.start() as u32, rec.end() as u32);
        println!("{} overlaps for {:#?}", result.len(), rec);
    }
}
