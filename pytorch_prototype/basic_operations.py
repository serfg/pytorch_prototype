import torch
from torch import nn
from typing import List, Dict

class CovLinear(torch.nn.Module):
    def __init__(self, in_shape, out_shape):
        super(CovLinear, self).__init__()
        self.in_shape = in_shape
        if type(out_shape) is dict:
            self.out_shape = out_shape
        else:
            self.out_shape = {}
            for key in self.in_shape.keys():
                self.out_shape[key] = out_shape
        if set(in_shape.keys()) != set(out_shape.keys()):
            raise ValueError("sets of keys of in_shape and out_shape must be the same")
            
        linears = {}
        for key in self.in_shape.keys():
            linears[key] = torch.nn.Linear(self.in_shape[key], 
                                           self.out_shape[key], bias = False)
        self.linears = nn.ModuleDict(linears)
        
    def forward(self, features : Dict[str, torch.Tensor]):
        
        for key in features.keys():
            if key not in self.linears.keys():
                raise ValueError(f"key {key} in the features was not present in the initialization")
                
        result = {}      
        for key, linear in self.linears.items():
            if key in features.keys():
                now = features[key].transpose(1, 2)
                now = linear(now)
                now = now.transpose(1, 2)
                result[key] = now
        return result
    
class CovCat(torch.nn.Module):
    def __init__(self):
        super(CovCat, self).__init__()
        
    def forward(self, covariants : List[Dict[str, torch.Tensor]]):
        result : Dict[str, List[torch.Tensor]] = {}
        for el in covariants:
            for key, tensor in el.items():
                if key in result.keys():
                    result[key].append(tensor)
                else:
                    result[key] = [tensor]
        ans : Dict[str, torch.Tensor] = {}
        for key in result.keys():
            ans[key] = torch.cat(result[key], dim = 1)
        
        return ans
    