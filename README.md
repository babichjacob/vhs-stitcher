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
Coming up with that model (neural network design) was easy, since it was pretty much identical to what we talked about in class, but implementing it in code provided between hours and days of challenges.

The first setup that I managed to get to work, practically unmodified from the [`thinc` library's intro tutorial](https://colab.research.google.com/github/explosion/thinc/blob/master/examples/00_intro_to_thinc.ipynb#scrollTo=liOTpmsYyxma&line=8&uniqifier=1) started at 9% accuracy because it looks at inputs as flat arrays instead of as 2D matrices. So, I tossed that setup and scoured `thinc`'s websites for examples of CNNs, but [the only one it has is for natural language processing (NLP)](https://github.com/explosion/thinc/blob/master/examples/03_pos_tagger_basic_cnn.ipynb), which is either totally unrelated to image processing CNNs or the two are deeply linked in a way that is beyond my understanding. 

Because I was not able to make a CNN with `thinc`'s provided layer types, I began to look in to [PyTorch](https://pytorch.org/), a more mature and mainstream library that has many more tutorials available and built-in support for 2D image processing, including pooling and convolutional filter layers. The greatest thing is that the only thing about my code I had to change was my model, because the `thinc` library includes a [PyTorchWrapper](https://thinc.ai/docs/usage-frameworks) that makes PyTorch models fully compatible with `thinc` training, testing, prediction, and storage (i.e. to and from disk). A first attempt ([adapted from this example](https://github.com/pytorch/examples/blob/master/mnist/main.py#L11)) brought the network to 83% accuracy, then some revisions (changing around parameters like number of convolutional filters and fully connected layer neurons) brought it to 86%, then again up to 90%. 

I chose to take a closer look at the predictions the network was making and saw that they were the **same for every input it got, as though it were completely blindly guessing**, making the accuracy measurements I was getting **meaningless**. I couldn't find out what was wrong with my code from looking at it, so I searched the internet for possible reasons: TODO, and TODO. 

TODO: Describe learning my program is useless because OBS removed the problem.

Once these more major problems were solved, another was able to come to light: the network gets incredibly overconfident quickly. This is fair because it's right 97% of the time, but I would still like more discretion and admittance of defeat for those remaining 3% of cases. To try to achieve this, I looked into modifying the loss function, which the neural network is always trying to minimize, to heavily penalize overconfident wrong answers (which results in rewarding unconfident answers). [This article](https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7) describes *mean square error loss* as (Yh-Y)² and "penalizes the model when it makes large mistakes and incentivizes small errors", but my testing showed too little change from before, so I upped the exponent from 2 to 4 (TODO: test result and comment on it).


#### Training and testing
TODO: loading, shuffling up equal and unequal within each epoch, how accuracy works, division of equal:unequal in each set.


<a name="combine"></a>

### Bringing it all together: automatically combine two VHS recordings with `stitch combine`
TODO: Would it be appropriate to include that Kronk meme here? Sure, but edit it out from the Canvas submission.

This is where all the work done with the previously described commands pays off. Once those commands have been run (and the accuracy of the network is satisfactory), **they never need to be run again**.


<a name="conclusion"></a>

## Conclusion
TODO: evaluate the program with that sample I recorded a couple nights ago

TODO: historically avoided AI / NNs / ML but feel quite empowered to continue down this path and apply it to other things, like my security system which was a rejected project topic


<a name="appendix"></a>

## Appendix

The source code associated with this project is included as a zip archive attached to the Canvas submission. 

It is also available to browse publicly on [GitHub here](https://github.com/babichjacob/vhs-stitcher).
