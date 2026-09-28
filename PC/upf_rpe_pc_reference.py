import math, time
import numpy as np

PI = math.pi

def limit_angle(a):
    a = float(a) % (2*PI)
    while a <= -PI: a += 2*PI
    while a > PI: a -= 2*PI
    return a

def rot4(h):
    c,s=math.cos(h),math.sin(h)
    T=np.eye(4); T[0,0]=c; T[0,1]=-s; T[1,0]=s; T[1,1]=c
    return T

def T5(t):
    T=np.eye(5); T[:4,:4]=rot4(t[3]); T[:4,4]=t[:4]; return T

def invT5(t):
    T=T5(t); R=T[:4,:4].T
    out=np.eye(5); out[:4,:4]=R; out[:4,4]=-R@t[:4]; return out

def state_T(T): return T[:4,4].copy()

def sph_to_cart(s):
    r,az,el=s
    return np.array([r*math.cos(az)*math.cos(el), r*math.sin(az)*math.cos(el), r*math.sin(el)],dtype=float)

def cart_to_sph(x):
    x=np.asarray(x); r=np.linalg.norm(x); az=math.atan2(x[1],x[0]); el=math.atan2(x[2],np.linalg.norm(x[:2])); return np.array([r,az,el])

def cholesky_upper(P):
    P=0.5*(P+P.T)
    P=P+np.eye(len(P))*1e-10
    return np.linalg.cholesky(P).T

def unscented_transform(sigmas,Wm,Wc,Q):
    x=Wm@sigmas
    d=sigmas-x
    P=d.T@(Wc[:,None]*d)+Q
    return x,0.5*(P+P.T)

class AlgebraicMethod4DoF:
    """Core 4-DoF algebraic method copied in structure from Yuri sir's repository."""
    def __init__(self,d0,x_ha=None,sigma_uwb=0.1):
        self.eps=[1.0]; self.d=[float(d0)]; self.x_ha_0=np.zeros(4) if x_ha is None else np.array(x_ha,float)
        self.x_ha_odom=np.zeros((1,4)); self.x_ca_odom=np.zeros((1,4))
        self.P_ha=np.zeros((1,4,4)); self.P_ca=np.zeros((1,4,4)); self.R=np.array([sigma_uwb**2])
        self.x_ca_0_alg=np.zeros(4); self.x_ca_0=np.zeros(4); self.x_ca_r=np.zeros(4); self.x_ha_r=np.zeros(4); self.x_ca_r_alg=np.zeros(4)
        self.horizon=10; self.wls_bool=False; self.debug_bool=False
    def get_update(self,d,dx_ha,dx_ca,q_ha=None,q_ca=None):
        q_ha=np.zeros((4,4)) if q_ha is None else np.asarray(q_ha,float); q_ca=np.zeros((4,4)) if q_ca is None else np.asarray(q_ca,float)
        c_ha=rot4(self.x_ha_odom[-1,-1]); c_ca=rot4(self.x_ca_odom[-1,-1])
        xh=self.x_ha_odom[-1]+c_ha@np.asarray(dx_ha,float); xc=self.x_ca_odom[-1]+c_ca@np.asarray(dx_ca,float)
        self.x_ha_odom=np.vstack((self.x_ha_odom,xh)); self.x_ca_odom=np.vstack((self.x_ca_odom,xc))
        self.P_ha=np.concatenate((self.P_ha,(self.P_ha[-1]+c_ha@q_ha@c_ha.T)[None,:,:])); self.P_ca=np.concatenate((self.P_ca,(self.P_ca[-1]+c_ca@q_ca@c_ca.T)[None,:,:]))
        e=.5*(self.d[0]**2 + xh[:3]@xh[:3] + xc[:3]@xc[:3] - float(d)**2)
        self.eps.append(float(e)); self.d.append(float(d))
        if len(self.eps)>=self.horizon:
            self.find_relative_pose(); self.trim_to_latest_measurements()
    def trim_to_latest_measurements(self):
        if len(self.eps)>self.horizon:
            self.eps.pop(1); self.d.pop(1); self.x_ha_odom=np.delete(self.x_ha_odom,1,axis=0); self.x_ca_odom=np.delete(self.x_ca_odom,1,axis=0); self.P_ha=np.delete(self.P_ha,1,axis=0); self.P_ca=np.delete(self.P_ca,1,axis=0)
    def find_relative_pose(self):
        A=np.zeros((1,6)); M=np.array([[1.,0.,1.]])
        for i in range(1,len(self.eps)):
            x2,y2,z2=self.x_ca_odom[i,:3]; x1=self.x_ha_odom[i,:3]
            Mk=x1.reshape(1,3)@np.array([[-x2,2*y2,x2],[-y2,-2*x2,y2],[z2,0,z2]])
            M=np.vstack((M,Mk)); A=np.vstack((A,np.hstack((x1,self.x_ca_odom[i,:3]))))
        mat=np.hstack((M,A)).astype(float); eps=np.asarray(self.eps)[:,None]
        try:
            x=np.linalg.lstsq(mat,eps,rcond=None)[0].ravel(); den=x[0]+x[2]
            if abs(den)<1e-12: return
            x=x/den
            if not np.isfinite(x).all() or x[2] < -1 or x[2] > 1: return
            x=np.r_[x[3:6],2*np.arccos(np.sqrt(np.clip(x[2],0,1)))]
            if np.isfinite(x).all(): self.x_ca_0_alg=x; self.wls_bool=True; self.calculate_relative_position()
        except np.linalg.LinAlgError: pass
    def calculate_relative_position(self):
        c0=rot4(self.x_ca_0_alg[-1])[:3,:3]; c1=rot4(self.x_ha_odom[-1,-1])[:3,:3]
        p=c1.T@(self.x_ca_0_alg[:3]+c0@self.x_ca_odom[-1,:3]-self.x_ha_odom[-1,:3])
        h=limit_angle(self.x_ca_0_alg[-1]+self.x_ca_odom[-1,-1]-self.x_ha_odom[-1,-1]); self.x_ca_r_alg=np.r_[p,h]

class ModifiedSigmaPoints:
    def __init__(self,n,alpha=1,beta=2,kappa=-1):
        self.n=n; self.alpha=alpha; self.beta=beta; self.kappa=kappa; lam=alpha**2*(n+kappa)-n
        c=.5/(n+lam); self.Wc=np.full(2*n+1,c); self.Wm=np.full(2*n+1,c); self.Wc[0]=lam/(n+lam)+(1-alpha**2+beta); self.Wm[0]=lam/(n+lam)
    def sigma_points(self,x,P):
        U=cholesky_upper((self.n+self.alpha**2*(self.n+self.kappa)-self.n)*P)
        U=U.copy(); caps={1:PI,2:PI/2,3:PI,7:PI,8:PI}
        for i,cap in caps.items(): U[i,i]=min(U[i,i],cap)
        S=np.zeros((2*self.n+1,self.n)); S[0]=x
        for k in range(self.n):
            S[k+1]=x+U[k]; S[self.n+k+1]=x-U[k]
        return S

class TargetTrackingUKF:
    N=9
    def __init__(self,x_ha_0=None,weight=1.,drift_correction_bool=True):
        self.t_oi_cij=np.zeros(4) if x_ha_0 is None else np.array(x_ha_0,float); self.t_oi_si_prev=self.t_oi_cij.copy(); self.t_oi_si=self.t_oi_cij.copy()
        self.Dt_sj=np.zeros(4); self.q_ca=np.zeros((4,4)); self.q_ha=np.zeros((4,4)); self.t_si_uwb=np.zeros(4); self.t_sj_uwb=np.zeros(4)
        self.weight=weight; self.drift_correction_bool=drift_correction_bool; self.dt=.1; self.alpha=1.; self.beta=2.; self.kappa=-1.; self.nlos_degradation=.9
        self.kf_x=np.zeros(9); self.kf_P=np.eye(9); self.Q=np.zeros((9,9)); self.R=np.array([[1.]])
        self.points=ModifiedSigmaPoints(9,1,2,-1); self.likelihood=np.finfo(float).tiny; self.t_si_sj=np.zeros(4); self.x_ha=self.t_oi_si.copy()
    def set_ukf_properties(self,kappa=-1,alpha=1,beta=2): self.kappa=kappa; self.alpha=alpha; self.beta=beta; self.points=ModifiedSigmaPoints(9,alpha,beta,kappa)
    def set_uwb_extrinsicity(self,a,b): self.t_si_uwb=np.array(a,float); self.t_sj_uwb=np.array(b,float)
    def calculate_initial_state(self,s,ca_heading): return np.array([s[0],s[1],s[2]],float)
    def set_initial_state(self,s,sigma_s,ca_heading,ca_sigma_heading,sigma_uwb):
        sc=self.calculate_initial_state(s,ca_heading); self.kf_x=np.array([sc[0],sc[1],sc[2],ca_heading,0,0,0,0,0.]); self.kf_P=np.diag([sigma_s[0]**2,sigma_s[1]**2,sigma_s[2]**2,ca_sigma_heading**2,1e-8,1e-8,1e-8,1e-8,1e-8]); self.calculate_x_ca()
    def set_host_agent_uncertainty(self,P):
        f=T5(self.t_oi_si_prev)[:4,:4]; self.q_ha=self.q_ha+f@P@f.T
    def predict(self,dx_ca,q=None):
        q=np.diag([1e-6,1e-6,1e-6,1e-8]) if q is None else np.asarray(q,float)
        f=T5(self.Dt_sj)[:4,:4]; self.Dt_sj=self.Dt_sj+f@np.asarray(dx_ca,float); self.q_ca=self.q_ca+f@q@f.T
    def fx(self,x):
        if self.drift_correction_bool:
            T_si_oi_k=invT5(self.t_oi_si_prev); T_oi_cij_k=T5(self.t_oi_cij); T_cij_cji_k=T5(np.r_[sph_to_cart(x[:3]),x[3]]); T_cji_sj_k=T5(x[4:-1]); T_si_sj_k=T_si_oi_k@T_oi_cij_k@T_cij_cji_k@T_cji_sj_k
            T_D=T5(np.array([0,0,0,x[-1]])); T_cij_oi=invT5(self.t_oi_cij); T_cji_cij_k=invT5(np.r_[sph_to_cart(x[:3]),x[3]]); T_oi_si_k=T5(self.t_oi_si_prev)
            T_cji_sj_d=T_cji_cij_k@T_cij_oi@T_oi_si_k@T_D@T_si_sj_k
        else: T_cji_sj_d=T5(x[4:-1])
        T_cji_sj=T_cji_sj_d@T5(self.Dt_sj); y=state_T(T_cji_sj); out=x.copy(); out[4:-1]=y
        if self.drift_correction_bool: out[-2]=out[-2]-out[-1]
        return out
    def hx(self,x):
        T_oi_cij=T5(self.t_oi_cij); T_cij_cji=T5(np.r_[sph_to_cart(x[:3]),x[3]]); T_cji_sj=T5(x[4:-1]); T_si_oi=invT5(self.t_oi_si); T_si_sj=T_si_oi@T_oi_cij@T_cij_cji@T_cji_sj; Tuw=invT5(self.t_si_uwb)@T_si_sj@T5(self.t_sj_uwb); return np.array([np.linalg.norm(state_T(Tuw)[:3])])
    def run_filter(self,dx_ca,q_ca,measurement,x_ha,P_x_ha,sigma_uwb,bool_drift=True):
        self.t_oi_si=np.array(x_ha,float); self.set_host_agent_uncertainty(P_x_ha); self.predict(dx_ca,q_ca)
        self.Q=np.zeros((9,9)); self.Q[4:-1,4:-1]=self.q_ca
        self._ukf_predict(); self.R=np.array([[np.linalg.norm(self.q_ha[:3,:3])**2+sigma_uwb**2]])
        self._ukf_update(np.array([measurement])); self.t_oi_si_prev=self.t_oi_si.copy(); self.Dt_sj=np.zeros(4); self.q_ca=np.zeros((4,4)); self.set_residual_drift(bool_drift); self.calculate_x_ca(); self.weight*=self.likelihood
    def _ukf_predict(self):
        S=self.points.sigma_points(self.kf_x,self.kf_P); F=np.array([self.fx(s) for s in S]); self.kf_x,self.kf_P=unscented_transform(F,self.points.Wm,self.points.Wc,self.Q)
    def _ukf_update(self,z):
        Sx=self.points.sigma_points(self.kf_x,self.kf_P); Z=np.array([self.hx(s) for s in Sx]); zp,Sc=unscented_transform(Z,self.points.Wm,self.points.Wc,self.R); Sc_inv=np.linalg.inv(Sc)
        dx=Sx-self.kf_x; dz=Z-zp; Pxz=np.zeros((9,1))
        for i in range(len(Sx)): Pxz += self.points.Wc[i]*np.outer(dx[i],dz[i])
        K=Pxz@Sc_inv; y=z-zp; self.kf_x=self.kf_x+(K@y).ravel(); self.kf_P=self.kf_P-K@Sc@K.T
        innov=float(y[0]); var=float(Sc[0,0]); self.likelihood=max(math.exp(-.5*innov*innov/var)/math.sqrt(2*PI*var),np.finfo(float).tiny)
        if abs(self.kf_x[2])>PI/2: self.kf_x[1]+=PI; self.kf_x[2]=PI-self.kf_x[2]
        self.kf_x[1]=limit_angle(self.kf_x[1]); self.kf_x[2]=limit_angle(self.kf_x[2]); self.kf_x[3]=limit_angle(self.kf_x[3]); self.kf_x[-1]=limit_angle(self.kf_x[-1]);
        self.kf_P[1,1]=min(self.kf_P[1,1],PI); self.kf_P[2,2]=min(self.kf_P[2,2],PI/2); self.kf_P[3,3]=min(self.kf_P[3,3],PI); self.kf_P[-1,-1]=min(self.kf_P[-1,-1],PI); self.kf_P[-2,-2]=min(self.kf_P[-2,-2],PI)
    def set_residual_drift(self,bool_drift=True):
        self.kf_P[:,-1]=0; self.kf_P[-1,:]=0; self.kf_P[-1,-1]=1e-20; self.kf_x[-1]=0
        if bool_drift:
            self.q_ca[:3,:3]+=self.q_ha[:3,:3]; self.kf_P[-1,-1]+=self.q_ha[-1,-1]; self.kf_P[-1,-1]=min(self.kf_P[-1,-1],PI**2)
        self.q_ha=np.zeros((4,4))
    def calculate_x_ca(self):
        T_oi_cij=T5(self.t_oi_cij); self.t_cij_cji=np.r_[sph_to_cart(self.kf_x[:3]),self.kf_x[3]]; T_cij_cji=T5(self.t_cij_cji); T_cji_sj=T5(self.kf_x[4:-1]); T_si_oi=invT5(self.t_oi_si); T_sj_sj=T5(self.Dt_sj); self.t_si_sj=state_T(T_si_oi@T_oi_cij@T_cij_cji@T_cji_sj@T_sj_sj)
    def copy(self):
        p=TargetTrackingUKF(self.t_oi_cij,self.weight,self.drift_correction_bool); p.set_ukf_properties(self.kappa,self.alpha,self.beta); p.kf_x=self.kf_x.copy(); p.kf_P=self.kf_P.copy(); p.Q=self.Q.copy(); p.R=self.R.copy(); p.Dt_sj=self.Dt_sj.copy(); p.q_ca=self.q_ca.copy(); p.x_ha=self.t_oi_si.copy(); p.t_oi_si_prev=self.t_oi_si_prev.copy(); p.q_ha=self.q_ha.copy(); p.t_si_sj=self.t_si_sj.copy(); p.likelihood=self.likelihood; return p

class UPFConnectedAgent:
    def __init__(self,x_ha_0=None,sigma_uwb_factor=1.,resample_factor=.1,drift_correction_bool=True):
        self.ha_x=np.zeros(4) if x_ha_0 is None else np.array(x_ha_0,float); self.ha_P=np.zeros((4,4)); self.particles=[]; self.totalWeight=1.; self.best_particle=None; self.n_altitude=self.n_azimuth=self.n_heading=0; self.sigma_uwb=0; self.sigma_uwb_factor=sigma_uwb_factor; self.resample_factor=resample_factor; self.drift_correction_bool=drift_correction_bool; self.t_si_uwb=np.zeros(4); self.t_sj_uwb=np.zeros(4); self.kappa=-1; self.alpha=1; self.beta=2
    def create_particle(self):
        p=TargetTrackingUKF(self.ha_x,1./(self.n_azimuth*self.n_altitude*self.n_heading),drift_correction_bool=self.drift_correction_bool); p.set_uwb_extrinsicity(self.t_si_uwb,self.t_sj_uwb); p.set_ukf_properties(self.kappa,self.alpha,self.beta); return p
    def split_sphere_in_equal_areas(self,r,sigma_uwb,n_altitude,n_azimuth,n_heading):
        self.n_altitude=n_altitude; self.n_azimuth=n_azimuth; self.n_heading=n_heading; self.sigma_uwb=self.sigma_uwb_factor*sigma_uwb
        if n_altitude%2==0: n_altitude+=1
        dalt=PI/n_altitude; alts=np.round([-PI/2+dalt/2+i*dalt for i in range(n_altitude)],4)
        surfaces=[(math.sin(a+dalt/2)-math.sin(a-dalt/2))/2 for a in alts]; area=surfaces[(len(surfaces)-1)//2]/n_azimuth; bins=[math.ceil(s/area) for s in surfaces]
        sigaz=[(2*PI/b)/math.sqrt(-8*math.log(.5)) for b in bins]; sigel=(PI/n_altitude)/math.sqrt(-8*math.log(.5)); heads=[-PI+(2*PI/n_heading)*j for j in range(n_heading)]; sigh=(2*PI/n_heading)/math.sqrt(-8*math.log(.5))
        for i,alt in enumerate(alts):
            azs=[-PI+(2*PI/bins[i])*j for j in range(bins[i])]
            for az in azs:
                s=np.array([r,az,alt]);
                for h in heads:
                    p=self.create_particle(); p.set_initial_state(s,[2*self.sigma_uwb,sigaz[i],sigel],h,sigh,self.sigma_uwb); self.particles.append(p)
        self.set_best_particle(self.particles[0])
    def set_best_particle(self,p): self.best_particle=p
    def compare_particle(self,a,b):
        e=b.t_si_sj-a.t_si_sj; return a is not b and np.linalg.norm(e[:3])<.1 and abs(e[-1])<.1
    def branch_kill_resampling(self):
        if not self.particles: raise RuntimeError('No particles left')
        new=[]; new_weights=[]; avg=self.resample_factor/len(self.particles)
        for p in self.particles:
            p.weight/=self.totalWeight
            weight=int(p.weight/avg)
            if weight>0:
                weight=2. if weight>1/self.resample_factor else 1.
                merged=False
                for i,kp in enumerate(new):
                    if self.compare_particle(kp,p): kp.weight+=weight; new_weights[i]+=p.likelihood; merged=True; break
                if not merged: p.weight=weight; new.append(p); new_weights.append(p.likelihood)
        self.particles=new; self.best_particle=self.particles[int(np.argmax(new_weights))]
    def run_model(self,dx_ca,measurement,q_ca=None,x_ha=None,P_x_ha=None):
        if not self.particles: raise RuntimeError('Initialize particles with split_sphere_in_equal_areas first')
        if x_ha is not None: self.ha_x=np.array(x_ha,float)
        if P_x_ha is not None: self.ha_P=np.array(P_x_ha,float)
        self.totalWeight=0.
        keep=[]
        for p in self.particles:
            try: p.run_filter(dx_ca,q_ca,measurement,self.ha_x,self.ha_P,self.sigma_uwb,self.drift_correction_bool); keep.append(p); self.totalWeight+=p.weight
            except np.linalg.LinAlgError: pass
        self.particles=keep; self.branch_kill_resampling()
    def estimate(self): return self.best_particle.t_si_sj.copy()

def local_increment(p0,p1,y0,y1):
    d=np.asarray(p1)-np.asarray(p0); c,s=math.cos(y0),math.sin(y0); local=np.array([c*d[0]+s*d[1],-s*d[0]+c*d[1],d[2],limit_angle(y1-y0)]); return local

def main():
    print('=== UPF-RPE PC REFERENCE IMPLEMENTATION ===')
    print('Core structure: 5 Algebraic 4-DoF + 5 independent UKF placeholders + 1 Yuri-sir-style UPF')
    print('UPF particle grid: n_altitude=2 -> 3 latitude bands, n_azimuth=4, n_heading=4 => 32 particles')
    dt=.05; steps=200; sigma_uwb=.01
    initial_h=np.array([0.,0.,1.,0.]); initial_c=np.array([3.5,1.,2.,.3]); initial_range=float(np.linalg.norm(initial_c[:3]-initial_h[:3])); alg=[AlgebraicMethod4DoF(initial_range) for _ in range(5)]
    upf=UPFConnectedAgent(x_ha_0=np.zeros(4),sigma_uwb_factor=1.,resample_factor=.1,drift_correction_bool=True)
    upf.split_sphere_in_equal_areas(initial_range,sigma_uwb,2,4,4)
    # The five peer UKFs are deliberately kept separate from the UPF. Yuri sir's repository does not define a
    # five-agent wrapper, so these are workload slots rather than a claim that five UKFs feed one UPF.
    peer=[TargetTrackingUKF(np.zeros(4),1.) for _ in range(5)]
    for p in peer:
        p.set_ukf_properties(-1,1,2); p.set_initial_state([initial_range,0,0],[2*sigma_uwb,.5,.5],0,.5,sigma_uwb)
    prev_h=initial_h.copy(); prev_c=initial_c.copy()
    t0=time.perf_counter(); finite=True; alg_valid=0
    for k in range(steps):
        t=(k+1)*dt
        h=np.array([2*math.cos(.3*t),2*math.sin(.3*t),1.,limit_angle(.3*t)])
        c=np.array([3.5+.4*math.sin(.2*t),1.+.2*math.cos(.2*t),2.+.2*math.sin(.17*t),.3+.1*math.sin(.15*t)])
        u_h=local_increment(prev_h[:3],h[:3],prev_h[3],h[3]); u_c=local_increment(prev_c[:3],c[:3],prev_c[3],c[3])
        rng=np.linalg.norm(c[:3]-h[:3])+np.random.default_rng(k).normal(0,sigma_uwb)
        P_h=np.diag([1e-4,1e-4,1e-4,1e-5])
        for i in range(5):
            d=float(np.linalg.norm(c[:3]-h[:3])+np.random.default_rng(1000+k+i).normal(0,sigma_uwb))
            alg[i].get_update(d,u_h,u_c,q_ha=P_h,q_ca=P_h)
            if alg[i].wls_bool: alg_valid+=1
            peer[i].run_filter(u_c,P_h,d,h,P_h,sigma_uwb,True)
        upf.ha_x=h; upf.ha_P=P_h; upf.run_model(u_c,rng,q_ca=P_h,x_ha=h,P_x_ha=P_h)
        prev_h=h; prev_c=c
        for a in alg:
            if not np.isfinite(a.x_ca_r_alg).all(): finite=False
        for p in peer+upf.particles:
            if not np.isfinite(p.kf_x).all() or not np.isfinite(p.kf_P).all() or not np.isfinite(p.weight): finite=False
        if k in (0,9,49,99,199): print(f'[step {k+1:3d}] UPF best = {upf.estimate()} | particles={len(upf.particles)}')
    elapsed=(time.perf_counter()-t0)*1000
    print('\n--- SUMMARY ---')
    print(f'Cycles: {steps} @ {1/dt:.1f} Hz'); print('Algebraic estimators: 5'); print('Peer UKFs: 5'); print(f'UPF particles: {len(upf.particles)}'); print(f'Total UKF objects: {5+len(upf.particles)}'); print(f'Average cycle time: {elapsed/steps:.3f} ms'); print(f'Algebraic valid-solution updates: {alg_valid}'); print('All states finite:', 'YES' if finite else 'NO')
    print('NOTE: This is a standalone PC reference of the repository core. ROS, plotting, logging, and hardware I/O are intentionally removed.')

if __name__=='__main__': main()
