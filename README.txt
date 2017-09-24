***** ITNPBD8 - Evolutionary & Heuristic Optimisation Assignment Code README *****

# -*- coding: utf-8 -*-

This program has been developed to read a RGB data file containing 1000 colours represented by arithmitic Red/Green/Blue colour co-ordinates, run three algorithms (Hill-climber, Iterated local search and Evolutionary Algorithms) for 30 runs.
The program initially imports the necessary libraries, then defines the functions used in this program and finally calls methods to execute the program functions and produce computational results.
The data file provided included 1000 colours, however as the analysis covered a set of 10, 100 and 1000 colours, three seperate RGB data files were created from the origional file provided - not this was done manually using excel.
Should the analysis wish to be run on a file of 10 colours (for example), the filename/directory must be ammended to point to the file to be run within the read_colours_file() function.

###

USER INSTRUCTIONS: 2527317 - ITNPGBD8 Assignment.py was developed in Python 3.6 using Jupyter Notebook (an exploratory data analysis tool) and may require a few additional libraries to the standard python libraries. 
To run the program it is advised to identify which libraries (imported at the beginning of the program) are not yet installed and install those in a preffered manner. Alternatively follow the instructions below:
1. Go to the Windows Command Prompt in Windows (->press windows_key on keyboard, ->type 'cmd' ->press enter_key on keyboard)

    a. Intall the Pandas library: (->type 'conda install anaconda', ->press enter_key on keyboard -> follow installation instructions 
       to install the Anaconda distribution which includes Pandas and its dependancies)
    
    b. Intall the Seaborn library: (->type 'pip install seaborn', ->press enter_key on keyboard -> follow installation instructions)

	(other libraries required can be installed in a simillar manner, however should be pre-installed with versions of Python 2.7 or the Anaconda Distribution (and above)

2. Once required libraries have been installed, re-open the program and run the code in your preferred python console - recommended 
   consoles include Eclipse, Spyder, or Jupyter (among others).

###

@author: David Haveron - Student Number 2527317

