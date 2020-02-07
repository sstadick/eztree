use bio::io::bed;
use fnv::FnvHashMap;
use ginterval::Interval;
use std::process;

type Tree<T> = FnvHashMap<String, EzTree<T>>;

struct EzTree<T: Default> {
    intervals: Vec<Interval<T>>,
    mask: usize,
}

// Heavily based off of https://github.com/jonhoo/ordsearch/blob/master/src/lib.rs
impl<T: Default> EzTree<T> {
    pub fn new(mut intervals: Vec<Interval<T>>) -> Self {
        intervals.sort();
        let n = intervals.len();
        let mut iter = intervals.into_iter();
        let mut v = Vec::with_capacity(n);
        Self::eytzinger_walk(&mut v, &mut iter, 0);

        // it's now safe to set the length, since all `n` elements have been inserted.
        unsafe { v.set_len(n) };

        // Maybe nightly only
        let mut mask = 1;
        while mask <= n {
            mask <<= 1;
        }
        mask -= 1;

        EzTree {
            intervals: v,
            mask: mask,
        }
    }

    // Insert items from the sorted iterator `iter` into `v` in complete binary tree order.
    //
    // Requires `iter` to be a sorted iterator. Requires v's capacity to be set to the number of
    // elements in `iter`. The length of `v` will not be changed by this function.
    fn eytzinger_walk<I>(v: &mut Vec<Interval<T>>, iter: &mut I, i: usize)
    where
        I: Iterator<Item = Interval<T>>,
    {
        if i >= v.capacity() {
            return;
        }
        // visit left child
        Self::eytzinger_walk(v, iter, 2 * i + 1);

        // put data at the root
        // we know the get_unchecked_mut and unwrap below are safe bc we set the Vec's cap to the
        // len of te iterator.
        *unsafe { v.get_unchecked_mut(i) } = iter.next().unwrap();

        // vist right child
        Self::eytzinger_walk(v, iter, 2 * i + 2);
    }

    // Find the first overlap position
    pub fn find_gte<'a>(&'a self, start: u32, stop: u32) -> Option<&'a Interval<T>> {
        use std::mem;
        let mut i = 0;

        // The finiky bit
        //
        // We want to prefetch a couple of levels down in the tree from where we are. However, we
        // can only fetch one cacheline at a time (assume a line holds 64b). We therefor need to
        // find at what depth a single prefetch fetches all the descendants. It turns out that, at
        // depth k under some node with index i, the leftmost child is at:
        //
        // 2^k * i + 2^(k-1) + 2^(k-2) + ... 2^0 = s^k * i + s^k - 1
        //
        // This follows from the fact that the leftmost immediate child of node i is at 2i + 1 by
        // recursivly expanding i. If your're curious the rightmost child is at:
        //
        // 2^k * i + 2^(k-1) + 2^(k-2) + ... + 2^0 = 2^k * i + 2^(k+1) - 1
        //
        // at depth k, there are 2^k children. We can fit 64/(sizeof(Interval<T>)) children in a
        // cacheline, so we want to use the depth k that is 64/sizeof(Interval<T>) children. So, we
        // want:
        //
        // 2^k = 64/sizeof(Interval<T>)
        //
        // But we don't actually *need* k, we only ever use 2^k. So, we can just use
        // 64/sizeof(Interval<T>) directly! We call this the multiplier.
        let multiplier = 64 / mem::size_of::<Interval<T>>();
        // now for those additions we had to do above. Well, we know that the offset is really just
        // 2^k = 1, and we know that multiplier == 2^k, so we're done right? Well sort of. The
        // prefetch instruction fetches the cacheline that *holds* the given memory address. Let's
        // denote cachelines with []. What if we have:
        //
        // [..., 2^k + 2^k-1] [2^k + s^k, ...]
        //
        // Essentially, we got unlucky with the alignment so that the leftmost child is not sharing
        // a cacheline with any of the other items at that level! That's not great. So, instead, we
        // prefetch the address that is half-way through the set of children. That way, we ensure
        // that we prefetch at least half of the items.
        let offset = multiplier + multiplier / 2;
        let _ = offset; // make nightly happy if we use it

        while i < self.intervals.len() {
            #[cfg(feature = "nightly")]
            // unsafe is safe because pointer is never dereferenced
            unsafe {
                use std::intrinsics::prefetch_read_data;
                prefetch_read_data(
                    self.items
                        .as_ptr()
                        .offset(((multiplier * i + offset) & self.mask) as isize),
                    3,
                )
            };

            // safe because i < self.intervals.len()
            i = if !unsafe { self.intervals.get_unchecked(i) }.overlap(start, stop) {
                2 * i + 1
            } else {
                2 * i + 2
            };
        }

        // we want ffs(~(i + 1))
        // since ctz(x) = ffs(x) - 1
        // we use ctz(~(i + 1)) + 1
        let j = (i + 1) >> ((!(i + 1)).trailing_zeros() + 1);
        if j == 0 {
            None
        } else {
            Some(unsafe { self.intervals.get_unchecked(j - 1) })
        }
    }
}

fn make_tree(input: &str) -> EzTree<usize> {
    let mut reader = bed::Reader::from_file(input).expect("Couldn't open the input file");
    let mut records = vec![];
    let mut intervals = vec![];
    let mut counter = 0;
    for record in reader.records() {
        let rec = record.ok().expect("Error Reading record.");
        let interval = Interval {
            start: rec.start() as u32,
            stop: rec.end() as u32,
            val: counter,
        };
        intervals.push(interval);
        records.push(rec);
    }
    EzTree {
        intervals: intervals,
        mask: 0,
    }
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
    }
}
