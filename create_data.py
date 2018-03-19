import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse as spa
from scipy.sparse.linalg import factorized,bicg,cg,lsqr
import time

#extent of fake data (pixels)
Dx = 1.0
Dy = 1.0
Dz = 1.0

#number of points in the volume
npts = int(1e5)

xvals = np.random.rand(npts)*Dx-Dx/2
yvals = np.random.rand(npts)*Dy-Dy/2
zvals = np.random.rand(npts)*Dz-Dz/2

#homogenous coordinates
r = np.vstack((xvals,yvals,zvals,np.ones_like(zvals)))

#some continuous changes for the fake data
def Mvar(r):
    #r=[x,y,z,1]
    Mscale = np.eye(4)
    Mscale[0,0] += 0.0000001*r[0]
    Mscale[1,1] += -0.0000001*r[1]
    Mscale[2,2] += 0.0000003*r[2]
    
    Mrotx = np.eye(4)
    thx = 0.08 + 0.02*np.sin(2.0*np.pi*r[0]/1.5)
    Mrotx[1,1] = Mrotx[2,2] = np.cos(thx)
    Mrotx[1,2] = -np.sin(thx)
    Mrotx[2,1] = np.sin(thx)

    Mroty = np.eye(4)
    thy = -0.05 + 0.07*np.cos(2.0*np.pi*r[1]/1.6)
    Mroty[0,0] = Mroty[2,2] = np.cos(thy)
    Mroty[0,2] = np.sin(thy)
    Mroty[2,0] = -np.sin(thy)

    Mrotz = np.eye(4)
    thz = 0.03 + 0.02*np.cos(2.0*np.pi*r[2]/1.8)
    Mrotz[0,0] = Mrotz[1,1] = np.cos(thz)
    Mrotz[0,1] = -np.sin(thz)
    Mrotz[1,0] = np.sin(thz)

    Mtrans = np.eye(4)
    Mtrans[0,3] = 0.25
    Mtrans[1,3] = -0.2
    Mtrans[2,3] = 0.3

    tform = Mtrans.dot(Mscale.dot(Mrotx.dot(Mroty.dot(Mrotz))))

    return tform.dot(r)


def makewireframe(r):
    #create 12 lines, npts each
    npts = 100
    rnew = np.zeros((4,npts*12))
    rnew[3,:] = 1.0
    xyz = []
    for i in np.arange(3):
       xyz.append([r[i,:].min(),r[i,:].max(),np.linspace(r[i,:].min(),r[i,:].max(),npts)])
    lines=0
    combs = []
    combs.append([0,0,2])
    combs.append([1,0,2])
    combs.append([0,1,2])
    combs.append([1,1,2])
    combs.append([0,2,0])
    combs.append([1,2,0])
    combs.append([0,2,1])
    combs.append([1,2,1])
    combs.append([2,0,0])
    combs.append([2,1,0])
    combs.append([2,0,1])
    combs.append([2,1,1])
    for c in combs:
        i1 = npts*lines
        i2 = npts*(lines+1)
        lines += 1
        rnew[0,i1:i2] = xyz[0][c[0]]
        rnew[1,i1:i2] = xyz[1][c[1]]
        rnew[2,i1:i2] = xyz[2][c[2]]
    return rnew

frame = makewireframe(r)
nf = frame.shape[1]
r = np.hstack((r,frame))

rnew= np.zeros_like(r)
for i in np.arange(r.shape[1]):
    rnew[:,i] = Mvar(r[:,i])

fig = plt.figure(2)
fig.clf()
ax = fig.add_subplot(231, projection='3d')
ax.scatter(r[0,-nf:-1],r[1,-nf:-1],r[2,-nf:-1],marker='.',color='b')
ax.scatter(rnew[0,-nf:-1],rnew[1,-nf:-1],rnew[2,-nf:-1],marker='.',color='r')
ax.set_aspect('equal')
ax.set_xlabel('x',fontsize=24)
ax.set_ylabel('y',fontsize=24)
ax.set_zlabel('z',fontsize=24)
plt.subplot(2,3,4)
diff = r-rnew
res = np.sqrt(np.power(diff[0,:],2)+np.power(diff[1,:],2)+np.power(diff[2,:],2))
plt.hist(res,bins=100,edgecolor='none')
xl = plt.gca().get_xlim()
plt.gca().set_xlim(0.001,1)
xl = plt.gca().get_xlim()
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.title('avg res = %0.3f mm'%res.mean())

f = open('test3d.txt','w')
for i in np.arange(r.shape[1]):
    f.write('%0.4f %0.4f %0.4f %0.4f %0.4f %0.4f\n'%(r[0,i],r[1,i],r[2,i],rnew[0,i],rnew[1,i],rnew[2,i]))
f.close()

data = np.loadtxt('test3d.txt')
npts = data.shape[0]
niters = [[1,1,1],[4,4,4]]
xnew = []
xnew.append(np.array([1,0,0,0]))
xnew.append(np.array([0,1,0,0]))
xnew.append(np.array([0,0,1,0]))
for niter in np.arange(len(niters)):
    #construct A
    ndiv = niters[niter]
    #divisions
    dx = np.linspace(data[:,0].min(),data[:,0].max(),ndiv[0]+1)
    dy = np.linspace(data[:,1].min(),data[:,1].max(),ndiv[1]+1)
    dz = np.linspace(data[:,2].min(),data[:,2].max(),ndiv[2]+1)
    dxn,dyn,dzn = np.meshgrid(dx,dy,dz,indexing='ij')
    
    xind = []
    yind = []
    zind = []
    xnodes = []
    ynodes = []
    znodes = []
    alpha= 0.0
    for i in np.arange(ndiv[0]):
        if i==(ndiv[0]-1):
            alpha=1.0
        xind.append((data[:,0] >= dx[i]) & (data[:,0] < dx[i+1]+alpha))
        alpha=0.0
        xnodes.append(np.zeros((len(dx),len(dy),len(dz))).astype('Bool'))
        xnodes[-1][i:+2,:,:] = True
    for i in np.arange(ndiv[1]):
        if i==(ndiv[1]-1):
            alpha=1.0
        yind.append((data[:,1] >= dy[i]) & (data[:,1] < dy[i+1]+alpha))
        alpha=0.0
        ynodes.append(np.zeros((len(dx),len(dy),len(dz))).astype('Bool'))
        ynodes[-1][:,i:i+2,:] = True
    for i in np.arange(ndiv[2]):
        if i==(ndiv[2]-1):
            alpha=1.0
        zind.append((data[:,2] >= dz[i]) & (data[:,2] < dz[i+1]+alpha))
        alpha=0.0
        znodes.append(np.zeros((len(dx),len(dy),len(dz))).astype('Bool'))
        znodes[-1][:,:,i:i+2] = True
    inds = []
    for ix in xind:
        for iy in yind:
            for iz in zind:
                inds.append(ix & iy & iz)
    nblocks = len(inds)
    bnodes = []
    if nblocks>1:
        for xn in xnodes:
            for yn in ynodes:
                for zn in znodes:
                    bnodes.append(xn & yn & zn)
    
    A = np.zeros((npts,nblocks*4))
    b = [data[:,0],data[:,1],data[:,2]] #rhs vectors
    #xnew = []
    #xnew.append(np.array([1.0,0,0,0]))
    #xnew.append(np.array([0,1.0,0,0]))
    #xnew.append(np.array([0,0,1.0,0]))
    x0 = []
    x0.append(np.tile(xnew[0],nblocks))
    x0.append(np.tile(xnew[1],nblocks))
    x0.append(np.tile(xnew[2],nblocks))
    
    nrows = 0
    for i in np.arange(nblocks):
        ind = inds[i]
        n = np.count_nonzero(ind)
        #A[(nrows+n*0):(nrows+n*1),0+i*4] = data[ind,3]
        #A[(nrows+n*0):(nrows+n*1),1+i*4] = data[ind,4]
        #A[(nrows+n*0):(nrows+n*1),2+i*4] = data[ind,5]
        #A[(nrows+n*0):(nrows+n*1),3+i*4] = 1.0*np.ones(n)
        A[ind,0+i*4] = data[ind,3]
        A[ind,1+i*4] = data[ind,4]
        A[ind,2+i*4] = data[ind,5]
        A[ind,3+i*4] = 1.0*np.ones(n)
        nrows+=n

    if nblocks>1:
        for i in np.arange(nblocks):
            for j in np.arange(i):
                nodes = bnodes[i] & bnodes[j]
                xextra = dxn[nodes]
                yextra = dyn[nodes]
                zextra = dzn[nodes]
                nextra = xextra.size
                extra = np.zeros((nextra,nblocks*4))
                extra[:,0+i*4] = xextra
                extra[:,1+i*4] = yextra
                extra[:,2+i*4] = zextra
                extra[:,3+i*4] = 1.0*np.ones(nextra)
                extra[:,0+j*4] = -1.0*xextra
                extra[:,1+j*4] = -1.0*yextra
                extra[:,2+j*4] = -1.0*zextra
                extra[:,3+j*4] = -1.0*np.ones(nextra)
                A = np.vstack((A,extra))
                for k in np.arange(len(b)):
                    b[k] = np.append(b[k],np.zeros(nextra))

    A = spa.csc_matrix(A)
    W = spa.eye(A.shape[0])*1e-3
    wd = W.diagonal()
    wd[nrows:] = 1e3
    W.setdiag(wd)
    ATW = A.transpose().dot(W)
    K = ATW.dot(A)
    if niter==0:
        lam = 1e-2
        ltf = 1e-15
    if niter==1:
        lam = 1e-6
        ltf = 1e8
    rmat = spa.eye(K.shape[0])*lam
    #rd = rmat.diagonal()
    #rd[3::4]*=ltf
    #rmat.setdiag(rd)
    K = K+rmat
    K = K.tocsc()
    t0=time.time()
    solve = factorized(K)
    print('factorization time: %0.1f sec'%(time.time()-t0))
    xnew = []
    Mstart = []
    Mnew = []
    Lm = []
    for i in np.arange(nblocks):
        Mnew.append(np.eye(4))
        Mstart.append(np.eye(4))
    for i in np.arange(len(x0)):
        Lm.append(ATW.dot(b[i])+rmat.dot(x0[i]))
        t0=time.time()
        xnew.append(solve(Lm[i]))
        print('solve time: %0.1f sec'%(time.time()-t0))
        #x,info = cg(K,ATW.dot(b[i]),x0=x0[i],tol=1e-3)
        #info = lsqr(A,b[i],x0=x0[i])
        #xnew.append(x)
        #perr = K.dot(xnew[i])-Lm
        #aerr = A.dot(xnew[i])-b[i]
        #for j in np.arange(nblocks):
        #    prec = np.linalg.norm(perr[j*4:(j+1)*4])/np.linalg.norm(Lm[j*4:(j+1)*4])
        #    acc = np.linalg.norm(aerr[j*4:(j+1)*4])
        #    print('solve %d block %d prec: %0.2e acc: %0.2e'%(i,j,prec,acc))
        #print('')
        for j in np.arange(nblocks):
            Mnew[j][i,:] = xnew[i][j*4:(j+1)*4]
            Mstart[j][i,:] = x0[i][j*4:(j+1)*4]
    print(xnew[0])
    print('niter = %d, nblocks = %d'%(niter,nblocks)) 
    if niter==0:
        for M in Mstart:
            print('\n'.join([''.join(['{:15.4f}'.format(item) for item in row]) for row in M]))
            print('')
    for i in np.arange(nblocks):
        print('x: %0.1e %0.1e'%(data[inds[i],0].min(),data[inds[i],0].max()))
        print('y: %0.1e %0.1e'%(data[inds[i],1].min(),data[inds[i],1].max()))
        print('z: %0.1e %0.1e'%(data[inds[i],2].min(),data[inds[i],2].max()))
        print('\n'.join([''.join(['{:15.4f}'.format(item) for item in row]) for row in Mnew[i]]))
        print('')
    
    n = data.shape[0]
    vdata = np.vstack((np.flipud(np.rot90(data[:,3:6])),np.ones(n)))
    dnew = np.zeros_like(vdata)
    for i in np.arange(nblocks):
        ind = inds[i]
        n = np.count_nonzero(ind)
        dnew[:,ind] = Mnew[i].dot(vdata[:,ind])
    #data[:,3] = dnew[0,:]
    #data[:,4] = dnew[1,:]
    #data[:,5] = dnew[2,:]
    
    ax = fig.add_subplot(2,3,2+niter, projection='3d')
    ax.scatter(r[0,-nf:-1],r[1,-nf:-1],r[2,-nf:-1],marker='.',color='b')
    ax.scatter(dnew[0,-nf:-1],dnew[1,-nf:-1],dnew[2,-nf:-1],marker='.',color='r')
    ax.set_aspect('equal')
    ax.set_xlabel('x',fontsize=24)
    ax.set_ylabel('y',fontsize=24)
    ax.set_zlabel('z',fontsize=24)
    plt.subplot(2,3,5+niter)
    diff = r-dnew
    print('diff.sum(): %e'%diff.sum())
    res = np.sqrt(np.power(diff[0,:],2)+np.power(diff[1,:],2)+np.power(diff[2,:],2))
    plt.hist(res,bins=100,edgecolor='none')
    plt.gca().set_xlim(xl)
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.title('avg res = %0.3f mm'%res.mean())
    
