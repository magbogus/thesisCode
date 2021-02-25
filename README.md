# Analysis and Comparison of sounds used for echolocation - thesis code


## Description
This code was created in order to analyse mouth clicks' recordings regarding amplitude and frequency. Basing on received data, it prepares plots for statistics.

## Installation
Before you run thesis code, you have to install libraries by entering to the command line:

```python
pip install matplotlib==3.3.3
pip install scipy==1.5.4
pip install numpy==1.19.3
```
There may be a need of using Python 3.9.0.

## Usage
In order to analyse particular recording, you have to set name of the wave file in line 20: 
```python
20   click_rec = wave.open('recordings\\recording_name.wav', 'r')
```
and then type in the console:

```python
python fourier.py
```
When you gather enough amount of data to prepare statistics, type in the console:

```python
python statistics.py
```