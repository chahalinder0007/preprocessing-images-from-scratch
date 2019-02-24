from pip._internal import main
from pip._internal.utils.misc import get_installed_distributions

required_pacs=["opencv-python","numpy","scipy","scikit-learn","pywavelets"]
installed=[x.key for x in get_installed_distributions()]
to_inst=[f for f in required_pacs if f not in installed]

for pacs in to_inst:
    print("installing package ::: %s \n"%pacs)
    main(["install",pacs])    
    
if len(to_inst)==0:
    print("requirements already satisfied")