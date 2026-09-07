
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

"""
Grids for the lid-driven cavity flow

Taken from:
- Lin, Wu, Zhang (2019), A mesh-free radial basis function-based 
  semi-Lagrangian lattice Boltzmann method for incompressible flows,
  Int J Numer Meth Fluids, 91, pg. 198-211

- Flyer, Bayona, ...

- Shu, ...

"""

r1 = 0.01
rmin, rmax = -0.5, 0.5

n = 101

points = np.empty((0, 2))

for i in range(11):

	t, h = np.linspace(rmin,rmax,num=n,retstep=True)
	print(i,h,rmin,rmax)

	# North
	xN = t
	yN = rmax*np.ones_like(t)

	# South
	xS = t
	yS = rmin*np.ones_like(t)

	# East
	xE = rmin*np.ones_like(t[1:-1])
	yE = t[1:-1]

	# West
	xW = rmax*np.ones_like(t[1:-1])
	yW = t[1:-1]

	rmin += r1
	rmax -= r1
	n -= 2

	xp = np.concatenate((xN, xS, xE, xW))
	yp = np.concatenate((yN, yS, yE, yW))
	tmp = np.stack((xp,yp),axis=1)
	points = np.vstack((points, tmp))


print(rmin,rmax)
rmin = -0.39
rmax =  0.39
r1 = 0.015
n = 26*2+1

for i in range(11):

	t, h = np.linspace(rmin,rmax,num=n,retstep=True)
	print(i,h,rmin,rmax)

	# North
	xN = t
	yN = rmax*np.ones_like(t)

	# South
	xS = t
	yS = rmin*np.ones_like(t)

	# East
	xE = rmin*np.ones_like(t[1:-1])
	yE = t[1:-1]

	# West
	xW = rmax*np.ones_like(t[1:-1])
	yW = t[1:-1]

	rmin += r1
	rmax -= r1
	n -= 2

	xp = np.concatenate((xN, xS, xE, xW))
	yp = np.concatenate((yN, yS, yE, yW))
	tmp = np.stack((xp,yp),axis=1)
	points = np.vstack((points, tmp))

r1 = 0.025
rmin = -0.25+r1
rmax =  0.25-r1
n = 9*2+1

for i in range(9):

	t, h = np.linspace(rmin,rmax,num=n,retstep=True)
	print(i,h,rmin,rmax)

	# North
	xN = t
	yN = rmax*np.ones_like(t)

	# South
	xS = t
	yS = rmin*np.ones_like(t)

	# East
	xE = rmin*np.ones_like(t[1:-1])
	yE = t[1:-1]

	# West
	xW = rmax*np.ones_like(t[1:-1])
	yW = t[1:-1]

	rmin += r1
	rmax -= r1
	n -= 2

	xp = np.concatenate((xN, xS, xE, xW))
	yp = np.concatenate((yN, yS, yE, yW))
	tmp = np.stack((xp,yp),axis=1)
	points = np.vstack((points, tmp))

points = np.vstack((points,np.array([[0,0]])))

plt.scatter(points[:,0],points[:,1])
plt.axis('equal')
plt.show()



h1 = 0.01
h2 = 0.015
h3 = 0.025

z1, h1 = np.linspace(-0.5,-0.4,11,retstep=True)
z2, h2 = np.linspace(-0.4,-0.25,11,retstep=True)
z3, h3 = np.linspace(-0.25,0,11,retstep=True)
print(h1,h2,h3)

z = np.concatenate((z1,z2[1:],z3[1:]))
z = np.concatenate((z[:-1],np.flip(z)))

zx, zy = np.meshgrid(z,z)
plt.scatter(zx,zy,marker='o')
plt.axis('equal')
plt.show()


t, r1 = np.linspace(-0.25,0.25,num=21, retstep=True)
xv,yv = np.meshgrid(t,t)

xp = xv[1:-1,1:-1].ravel()
yp = yv[1:-1,1:-1].ravel()


t, r2 = np.linspace(-0.4,-0.25,num=10, retstep=True)
x2v,y2v = np.meshgrid(t,t)

print(r1, r2)

plt.scatter(xv,yv,marker='o')
plt.scatter(x2v,y2v,marker='s')

plt.axis("equal")

plt.show()



