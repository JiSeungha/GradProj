import os

path = str(os.getcwd()) + '/Export'
input_file = "RKneeY20_i.txt"
output_file = "RKneeY20_o.txt"
inF = open(input_file,'w')
outF = open(output_file,'w')
files = os.listdir(path)

for file in files:
	if "rki.txt" not in file:
		continue
	else: 
		file_ = open(path +'/' + file)
		for line in file_:
			inF.write(line)
		file_.close()

for file in files:
	if "rko.txt" not in file:
		continue
	else :
		file_ = open(path +'/' + file)
		for line in file_:
			outF.write(line) 
		file_.close()

inF.close()
outF.close()