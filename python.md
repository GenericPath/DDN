Create a [conda](https://www.anaconda.com/) environment
```python
conda create --name ddn python=3.7
```
Install pytorch (based on best available CUDA version)
e.g. 
```python
conda install pytorch torchvision torchaudio cudatoolkit=*10.1* -c pytorch
```
then install additional requirements
```python
pip install -r requirements.txt
```
If wishing to use GPU, you may have to uninstall cpuonly pytorch via
```python
conda uninstall cpuonly
```
and this will enable GPU usage

---
windows may need additional library
```python
conda install pywin32
```
