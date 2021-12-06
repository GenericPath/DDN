Deep Declarative Networks - Summary Research Bursary
---

Install pytorch (based on best available CUDA version)
e.g. 
> conda install pytorch torchvision torchaudio cudatoolkit=*10.1* -c pytorch

then install additional requirements
> pip install -r requirements.txt

If cpuonly version installed, then simply uninstall via
> conda uninstall cpuonly
and this will enable GPU usage

