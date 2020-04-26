import math


def sindeg(angle):
    return math.sin(angle/180*math.pi)

def cosdeg(angle):
    return math.cos(angle/180*math.pi)

def tandeg(angle):
    return math.tan(angle/180*math.pi)

def rtsmsq(n1,n2):
    return (n1**2+n2**2)**0.5

dist = 2.8   # light's center to focal-plane's center
angy = 81   # angle to y axis
angz = 89   # angle to x axis
scale = 0.1 # meter

yedge = 0.5881 # meter
zedge = 0.5 # meter


dist_v = dist/(1+(cosdeg(angy)**2/(1-cosdeg(angy)**2)+(cosdeg(angy)**2/(1-cosdeg(angy)**2))))**0.5
print('Vertical distance:', dist_v)

y = dist_v*cosdeg(angy)/(1-cosdeg(angy)**2)**0.5
z = dist_v*cosdeg(angz)/(1-cosdeg(angz)**2)**0.5

dlong = rtsmsq(z+zedge/2, rtsmsq(y+yedge/2,dist))
dshort = rtsmsq(z+zedge/2-scale, rtsmsq(y+yedge/2-scale,dist))
dmean = rtsmsq(z+zedge/2-scale/2, rtsmsq(y+yedge/2-scale/2,dist))

ilnu = (1/dshort**2-1/dlong**2)/(1/dmean**2)

print('D_long  D_short  ENU(%):')
print('{0:6.2f}'.format(dlong), '{0:7.2f}'.format(dshort), '{0:7.2%}'.format(ilnu))

