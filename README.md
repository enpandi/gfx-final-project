# Extra Credit (10 points)

I affirm, on my honor, that I have completed the online course instructor survey (both C S 354H and C S 384P)

# Running the code

install rust if you haven't already: https://www.rust-lang.org/learn/get-started

I developed using rustc 1.77.2/Rust 2021, but it shouldn't affect compatibility too much

`cargo run --release` to run the app

in ./Cargo.toml, comment out 'default = ["four_d"]' to switch back to 3d rendering/simulation


Run geometric-algebra/codegen/gen.py to generate new algebra libraries.



# Controls
Space                => go up
Left Shift           => go down
W                    => go forward
S                    => go backward
A                    => go left
D                    => go right
Enter                => hold key to enable physics
Mouse                => camera
Mouse 0 (left click) => place an object in front of you (random box/sphere)
Mouse 3 (back)       => rotate camera along the look axis
Mouse 4 (forward)    => rotate camera along the look axis
Mouse scroll         => camera zoom (untested)
