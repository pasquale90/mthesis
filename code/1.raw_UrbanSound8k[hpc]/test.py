import console

m="a"
v=15
console_path = console.makefile(m,v)
print("fasdgasd", file=open(console_path, "a"))
                                              
print(type(console_path))
print(console_path)