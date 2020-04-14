# 📼 VHS Stitcher

This application is being created for my Robot Vision class, ECE 4881 at the University of Michigan-Dearborn.


<a name="report"></a>

# 📄 Report


<a name="intro"></a>

## Introduction: project goals
TODO


<a name="implementation"></a>

## Implementation and application usage
This project is written in version 3 of Python only.


### Creating a command line interface
First, every feature of the application, which will be described in upcoming sections, has its own Python file: `extract.py`, `match.py`, `assemble.py`, `train.py`, and `combine.py`. This is just good programming practice (to break up semi-unrelated things in to different files), but doing this also made it easy to run each file as its own program on the command line, using a library from Google called [`fire`](https://github.com/google/python-fire).

For instance, `python extract.py video1.mp4 video2.mp4` calls the `main` function in `extract.py` with arguments `chunk_1="video1.mp4"` and `chunk_2="video2.mp4"`. 

For my taste, this could still be improved upon, so I made the `stitch` (macOS and Linux) and `stitch.cmd` (Windows) scripts to wrap the Python command conveniently. These make the above command equivalent to `./stitch extract video1.mp4 video2.mp4`.


<a name="extract"></a>

### Extracting frames from the end of one chunk and start of another: `stitch extract`
TODO


<a name="match"></a>

### Manually labeling frame data: `stitch match`
TODO


<a name="assemble"></a>

### Assembling training and test sets: `stitch assemble`
TODO


<a name="train"></a>

### Training the neural network: `stitch train`
#### Convolutional neural network (CNN) architecture
TODO

Include that diagram (modified) I found on that one site here


#### Adversity
TODO
Describe how I thought my neural network worked with great (90%) accuracy until I took a closer look to see that it gave the same outputs for every input, as though it couldn't even see them. 

Describe learning my program is useless because OBS removed the problem.

Once these more major problems were solved, another was able to come to light: the network gets incredibly overconfident quickly. This is fair because it's right 97% of the time, but I would still like more discretion and admittance of defeat for those remaining 3% of cases. To try to achieve this, I looked into modifying the loss function, which the neural network is always trying to minimize, to heavily penalize overconfident wrong answers (which results in rewarding unconfident answers). [This article](https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7) describes *mean square error loss* as (Yh-Y)² and "penalizes the model when it makes large mistakes and incentivizes small errors", but my testing showed too little change from before, so I upped the exponent from 2 to 4 (TODO: test result and comment on it).


<a name="combine"></a>

### Bringing it all together: automatically combine two VHS recordings with `stitch combine`
Would it be appropriate to include that Kronk meme here? Sure, but edit it out from the Canvas submission.

This is where all the work done with the previously described commands from the previous sections pays off. Once those commands have been run (and the accuracy of the network is satisfactory), **they never need to be run again**.


<a name="conclusion"></a>

## Conclusion
TODO