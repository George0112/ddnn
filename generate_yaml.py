import sys
import argparse
from template.app_yaml import *

parser = argparse.ArgumentParser()

model_choices = ['vgg16', 'multitask']

parser.add_argument("--model", type=str, required="True", choices=model_choices)
parser.add_argument("--cut-point", nargs="+", default=[0, -1])
parser.add_argument("--output-layer", nargs="+", default=[-1], type=int)
parser.add_argument("--input-size", type=int, default=224)

args = parser.parse_args()

max_layer = 200

if(int(args.cut_point[-1]) > max_layer):
  print("max layer = %s" %(max_layer))
  sys.exit()

m = args.model
flag = '"--is-first"'
app = m + '-0'
c = 0
n = args.cut_point[0]

def write():
  with open("./yaml/%s-svc.yaml" %(app), "w") as f:
    f.write(svc_yaml %(app, app, app))

  with open("./yaml/%s.yaml" %(app), "a") as f:
    f.write(app_yaml %(app, app, app, app, m, m, c, n, flag))

write()
flag = ''

for idx, i in enumerate(args.cut_point):
  c = int(args.cut_point[idx])+1 # The current cut point
  if(idx == len(args.cut_point)-2):
    app = m + '-' + str(c)
    if(args.output_layer[0] != -1):
      n = args.cut_point[idx+1] + '", "' + '", "'.join(str(o) for o in args.output_layer) # Multiple outputs
    else:
      n = args.cut_point[idx+1]

    write()

  elif(idx == len(args.cut_point)-1):
    print(args.output_layer)
    flag = '"--is-last"'
    if(args.output_layer[0] == -1):
      app=m +'-'+ str(int(args.cut_point[-1])+1)
      n = -1 # Only 1 output
      write()
    else:
      for o in args.output_layer:
        if o == -1:
          break
        n = o
        app = m+'-'+str(o)
        write()
  else:
    app=m +'-'+ str(c)
    n = args.cut_point[idx+1]
    write()

  flag = ""

with open("./yaml/%s-client.yaml" %(m), "w") as f:
  f.write(client_yaml %(m, m, m, m, m, m, args.input_size))