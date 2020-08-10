# MRes-2020DS-Project2
This repository contains codes used and relative documentation for the 2nd project of Imperial College London MRes Biomedical Research (Data Science stream) in 2019-2020 academic year. Project thesis will be uploaded in due course.\
\
-- Conducted by: Mingze Gao.\
-- Supervisors: Dr. Joram Matthias Posma & Dr. Katia De Filippo

## Project Aims & Objectives
The overarching aim of this project is to implement ML/DL based algorithms ([fastER](https://bsse.ethz.ch/csd/software/faster.html) / [Mask-RCNN](https://github.com/mirzaevinom/data_science_bowl_2018)) to,
1. Analyse splenic IVM video data
2. Compare **cell segmentation performance** with [Imaris](https://imaris.oxinst.com/)
3. Use statistical metrics to quantify the **migration characteristics** of neutrophils
4. Investigate **Neutrophil-B cell interaction behaviours** under different experimental conditions.

## Platform & Codes
All codes are implemented and tested in Python 3.7.3 and Conda 4.8.3\
Programs are ran locally on Windows 10, Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz, RAM 8.00GB.\
Deep learning codes are connected to HPC based on NVIDIA Tesla K80 GPU with 24GB memory.\
\
Multiple object tracking is based on [SORT](https://github.com/abewley/sort), codes are revised and acknowledged approriately inside thesis to fulfill project needs.\
\
Uploaded codes contains all the scripts in this project ranging from, data pre-processing, extraction, cleaning, cell trajectory tracking, migration and interaction analysis, to reproducible data visualisation (raw data excluded due to the nature of the experiment).
