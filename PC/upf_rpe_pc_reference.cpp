#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#ifdef NSIG
#undef NSIG
#endif

using std::array;
using std::vector;
constexpr double PI = 3.1415926535897932384626433832795;

static double limit_angle(double a){
    a=std::fmod(a+PI,2*PI); if(a<0) a+=2*PI; return a-PI;
}

struct V4{ array<double,4> a{}; double& operator[](int i){return a[i];} double operator[](int i)const{return a[i];} };
struct V9{ array<double,9> a{}; double& operator[](int i){return a[i];} double operator[](int i)const{return a[i];} };
struct M4{ double a[4][4]{}; double& operator()(int r,int c){return a[r][c];} double operator()(int r,int c)const{return a[r][c];} };
struct M5{ double a[5][5]{}; double& operator()(int r,int c){return a[r][c];} double operator()(int r,int c)const{return a[r][c];} };
struct M9{ double a[9][9]{}; double& operator()(int r,int c){return a[r][c];} double operator()(int r,int c)const{return a[r][c];} };

static M4 rot4(double h){M4 T{}; for(int i=0;i<4;i++)T(i,i)=1; double c=std::cos(h),s=std::sin(h); T(0,0)=c;T(0,1)=-s;T(1,0)=s;T(1,1)=c;return T;}
static M5 T5(const V4&t){M5 T{};for(int i=0;i<5;i++)T(i,i)=1;M4 R=rot4(t[3]);for(int r=0;r<4;r++)for(int c=0;c<4;c++)T(r,c)=R(r,c);for(int r=0;r<4;r++)T(r,4)=t[r];return T;}
static M5 mul(const M5&A,const M5&B){M5 C{};for(int i=0;i<5;i++)for(int j=0;j<5;j++)for(int k=0;k<5;k++)C(i,j)+=A(i,k)*B(k,j);return C;}
static M5 invT5(const V4&t){M5 T=T5(t),O{};for(int i=0;i<5;i++)O(i,i)=1;for(int r=0;r<4;r++)for(int c=0;c<4;c++)O(r,c)=T(c,r);for(int r=0;r<4;r++){O(r,4)=0;for(int c=0;c<4;c++)O(r,4)-=O(r,c)*t[c];}return O;}
static V4 stateT(const M5&T){V4 v;for(int i=0;i<4;i++)v[i]=T(i,4);return v;}
static array<double,3> sph(const V4&s){return {s[0]*std::cos(s[1])*std::cos(s[2]),s[0]*std::sin(s[1])*std::cos(s[2]),s[0]*std::sin(s[2])};}
static V4 cartSph(const array<double,3>&p){V4 s;double r=std::sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);s[0]=r;s[1]=std::atan2(p[1],p[0]);s[2]=std::atan2(p[2],std::hypot(p[0],p[1]));return s;}

static bool finiteV4(const V4&v){for(double x:v.a)if(!std::isfinite(x))return false;return true;}
static bool finiteV9(const V9&v){for(double x:v.a)if(!std::isfinite(x))return false;return true;}
static bool finiteM9(const M9&M){for(int i=0;i<9;i++)for(int j=0;j<9;j++)if(!std::isfinite(M(i,j)))return false;return true;}

static bool cholUpper9(const M9&P,M9&U){U=M9{};for(int i=0;i<9;i++)for(int j=i;j<9;j++){double s=P(i,j);for(int k=0;k<i;k++)s-=U(k,i)*U(k,j);if(i==j){if(s<=1e-10)s=1e-10;U(i,j)=std::sqrt(s);}else{if(std::abs(U(i,i))<1e-14)return false;U(i,j)=s/U(i,i);}}return finiteM9(U);}

static bool solveLinear(vector<vector<double>> A,vector<double>b,vector<double>&x){int n=(int)b.size();x.assign(n,0);for(int c=0;c<n;c++){int p=c;for(int r=c+1;r<n;r++)if(std::abs(A[r][c])>std::abs(A[p][c]))p=r;if(std::abs(A[p][c])<1e-12)return false;std::swap(A[p],A[c]);std::swap(b[p],b[c]);double q=A[c][c];for(int j=c;j<n;j++)A[c][j]/=q;b[c]/=q;for(int r=0;r<n;r++)if(r!=c){double f=A[r][c];for(int j=c;j<n;j++)A[r][j]-=f*A[c][j];b[r]-=f*b[c];}}x=b;return true;}

class AlgebraicMethod4DoF{
public:
    static constexpr int H=10; double d0; vector<double> eps,d; array<V4,H> xh{},xc{}; int n=1; V4 x0{},xr{}; bool valid=false;
    AlgebraicMethod4DoF(double d0_):d0(d0){eps={1};d={d0};}
    void update(double meas,const V4&dxh,const V4&dxc){V4 hp=xh[n-1],cp=xc[n-1],hc=hp,cc=cp;M4 Rh=rot4(hp[3]),Rc=rot4(cp[3]);for(int i=0;i<3;i++){hc[i]+=Rh(i,0)*dxh[0]+Rh(i,1)*dxh[1]+Rh(i,2)*dxh[2];cc[i]+=Rc(i,0)*dxc[0]+Rc(i,1)*dxc[1]+Rc(i,2)*dxc[2];}hc[3]=limit_angle(hp[3]+dxh[3]);cc[3]=limit_angle(cp[3]+dxc[3]);double hn=hc[0]*hc[0]+hc[1]*hc[1]+hc[2]*hc[2],cn=cc[0]*cc[0]+cc[1]*cc[1]+cc[2]*cc[2];double e=.5*(d0*d0+hn+cn-meas*meas);if(n<H){xh[n]=hc;xc[n]=cc;eps.push_back(e);d.push_back(meas);n++;}else{for(int i=1;i<H;i++){xh[i-1]=xh[i];xc[i-1]=xc[i];eps[i-1]=eps[i];d[i-1]=d[i];}xh[H-1]=hc;xc[H-1]=cc;eps[H-1]=e;d[H-1]=meas;}if(n>=H){solve();trim();}}
    void solve(){vector<vector<double>>A(9,vector<double>(6));vector<double>Y(9);for(int i=1;i<H;i++){double x=xc[i][0],y=xc[i][1],z=xc[i][2];double X=xh[i][0],Yh=xh[i][1],Z=xh[i][2];A[i-1][0]=1;A[i-1][1]=0;A[i-1][2]=1;A[i-1][3]=X;A[i-1][4]=Yh;A[i-1][5]=Z;A[i-1][3]+=x;A[i-1][4]+=y;A[i-1][5]+=z;Y[i-1]=eps[i];}
        // The repository uses np.linalg.lstsq on a 9x6 matrix. For a dependency-free C++ PC port we solve the equivalent normal equations.
        vector<vector<double>>N(6,vector<double>(6));vector<double>b(6);for(int r=0;r<9;r++)for(int i=0;i<6;i++){b[i]+=A[r][i]*Y[r];for(int j=0;j<6;j++)N[i][j]+=A[r][i]*A[r][j];}for(int i=0;i<6;i++)N[i][i]+=1e-10;vector<double>x;if(!solveLinear(N,b,x))return;double den=x[0]+x[2];if(std::abs(den)<1e-12)return;for(double&q:x)q/=den;if(x[2]<-1||x[2]>1)return;array<double,4> q{x[3],x[4],x[5],2*std::acos(std::sqrt(std::clamp(x[2],0.0,1.0)))};for(double qv:q)if(!std::isfinite(qv))return;for(int i=0;i<4;i++)x0[i]=q[i];M4 C0=rot4(x0[3]),C1=rot4(xh[H-1][3]);array<double,3> v{};for(int r=0;r<3;r++){double zr=x0[r]+C0(r,0)*xc[H-1][0]+C0(r,1)*xc[H-1][1]+C0(r,2)*xc[H-1][2]-xh[H-1][r];v[0]=zr;v[1]=zr;v[2]=zr;xr[r]=C1(0,r)*v[0]+C1(1,r)*v[1]+C1(2,r)*v[2];}xr[3]=limit_angle(x0[3]+xc[H-1][3]-xh[H-1][3]);valid=finiteV4(xr);}
    void trim(){if((int)eps.size()>H){eps.erase(eps.begin()+1);d.erase(d.begin()+1);for(int i=1;i<H;i++){xh[i-1]=xh[i];xc[i-1]=xc[i];}}}
};

class UKFParticle{
public:
    static constexpr int L=9,S=19;V9 x{};M9 P{};double weight=1,likelihood=std::numeric_limits<double>::min();V4 Dt{},t_oi_si_prev{},t_oi_si{},t_oi_cij{},t_si_uwb{},t_sj_uwb{};M9 Q{};double Wm[S]{},Wc[S]{};double alpha=1,beta=2,kappa=-1;M4 qca{};
    UKFParticle(const V4&host=V4{},double w=1):weight(w),t_oi_si_prev(host),t_oi_si(host),t_oi_cij(host){double lam=alpha*alpha*(L+kappa)-L;Wm[0]=lam/(L+lam);Wc[0]=Wm[0]+(1-alpha*alpha+beta);for(int i=1;i<S;i++)Wm[i]=Wc[i]=1.0/(2*(L+lam));for(int i=0;i<9;i++)P(i,i)=1;}
    void init(const V4&s,double sr,double saz,double sel,double heading,double sh){x=V9{};x[0]=s[0];x[1]=s[1];x[2]=s[2];x[3]=heading;P=M9{};P(0,0)=sr*sr;P(1,1)=saz*saz;P(2,2)=sel*sel;P(3,3)=sh*sh;for(int i=4;i<8;i++)P(i,i)=1e-8;P(8,8)=1e-8;}
    V4 ca_state()const{V4 v;for(int i=0;i<4;i++)v[i]=x[4+i];return v;}
    void predictOdom(const V4&dx,const M4&q){M5 F=T5(Dt);M4 f{};for(int r=0;r<4;r++)for(int c=0;c<4;c++)f(r,c)=F(r,c);V4 y;for(int r=0;r<4;r++){y[r]=Dt[r];for(int c=0;c<4;c++)y[r]+=f(r,c)*dx[c];}Dt=y;for(int r=0;r<4;r++)for(int c=0;c<4;c++)qca(r,c)+=sumF(f,q,r,c);}
    static double sumF(const M4&f,const M4&q,int r,int c){double s=0;for(int i=0;i<4;i++)for(int j=0;j<4;j++)s+=f(r,i)*q(i,j)*f(c,j);return s;}
    V9 fx(const V9&in)const{V4 xv;for(int i=0;i<4;i++)xv[i]=in[4+i];M5 outT;if(true){M5 A=invT5(t_oi_si_prev),B=T5(t_oi_cij),C;V4 sc;auto p=sph({in[0],in[1],in[2],in[3]});sc[0]=p[0];sc[1]=p[1];sc[2]=p[2];sc[3]=in[3];C=T5(sc);M5 D=T5(xv);M5 TS=mul(mul(mul(A,B),C),D);V4 drift;drift[0]=0;drift[1]=0;drift[2]=0;drift[3]=in[8];M5 TD=T5(drift);M5 Tci=invT5(sc),Toc=invT5(t_oi_cij),Tosi=T5(t_oi_si_prev);outT=mul(mul(mul(mul(Tci,Toc),Tosi),TD),TS);}M5 nd=mul(outT,T5(Dt));V4 y=stateT(nd);V9 out=in;for(int i=0;i<4;i++)out[4+i]=y[i];out[7]=out[7]-out[8];return out;}
    double hx(const V9&in)const{V4 sc;auto p=sph({in[0],in[1],in[2],in[3]});sc[0]=p[0];sc[1]=p[1];sc[2]=p[2];sc[3]=in[3];V4 od;for(int i=0;i<4;i++)od[i]=in[4+i];M5 T=mul(mul(mul(invT5(t_oi_si),T5(t_oi_cij)),T5(sc)),T5(od));T=mul(mul(invT5(t_si_uwb),T),T5(t_sj_uwb));V4 v=stateT(T);return std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);}
    void sigma(V9&center,M9&PP,V9 out[S])const{M9 U{};cholUpper9(PP,U);double gamma=std::sqrt(L+kappa+alpha*alpha*(L+kappa)-L); // sqrt(n+lambda)
        double lam=alpha*alpha*(L+kappa)-L;gamma=std::sqrt(L+lam);out[0]=center;for(int k=0;k<L;k++){for(int r=0;r<L;r++){out[k+1][r]=center[r]+U(k,r)*gamma;out[k+1+L][r]=center[r]-U(k,r)*gamma;}}}
    void predictUKF(){V9 Sg[S],F[S];sigma(x,P,Sg);for(int i=0;i<S;i++)F[i]=fx(Sg[i]);V9 nx{};for(int i=0;i<S;i++)for(int r=0;r<L;r++)nx[r]+=Wm[i]*F[i][r];M9 NP{};for(int i=0;i<S;i++)for(int r=0;r<L;r++)for(int c=0;c<L;c++)NP(r,c)+=Wc[i]*(F[i][r]-nx[r])*(F[i][c]-nx[c]);for(int i=0;i<9;i++)for(int j=0;j<9;j++)Q(i,j)=0;for(int r=4;r<8;r++)for(int c=4;c<8;c++)Q(r,c)=qca(r-4,c-4);for(int r=0;r<9;r++)for(int c=0;c<9;c++)NP(r,c)+=Q(r,c);x=nx;P=NP;}
    void update(double z,double R){V9 Sg[S];sigma(x,P,Sg);double zp=0,Z[S];for(int i=0;i<S;i++){Z[i]=hx(Sg[i]);zp+=Wm[i]*Z[i];}double Sc=R;for(int i=0;i<S;i++)Sc+=Wc[i]*(Z[i]-zp)*(Z[i]-zp);if(Sc<1e-12)Sc=1e-12;array<double,9> Pxz{};for(int i=0;i<S;i++)for(int r=0;r<L;r++)Pxz[r]+=Wc[i]*(Sg[i][r]-x[r])*(Z[i]-zp);double innov=z-zp;for(int r=0;r<L;r++)x[r]+=Pxz[r]/Sc*innov;for(int r=0;r<L;r++)for(int c=0;c<L;c++)P(r,c)-=Pxz[r]*Pxz[c]/Sc;likelihood=std::max(std::exp(-.5*innov*innov/Sc)/std::sqrt(2*PI*Sc),std::numeric_limits<double>::min());if(std::abs(x[2])>PI/2){x[1]+=PI;x[2]=PI-x[2];}x[1]=limit_angle(x[1]);x[2]=limit_angle(x[2]);x[3]=limit_angle(x[3]);x[8]=limit_angle(x[8]);P(1,1)=std::min(P(1,1),PI);P(2,2)=std::min(P(2,2),PI/2);P(3,3)=std::min(P(3,3),PI);P(7,7)=std::min(P(7,7),PI);P(8,8)=std::min(P(8,8),PI);}
    void resetDrift(const M4&pha){for(int i=0;i<9;i++){P(i,8)=0;P(8,i)=0;}P(8,8)=1e-20;x[8]=0;for(int r=0;r<3;r++)for(int c=0;c<3;c++)qca(r,c)+=pha(r,c);qca(3,3)+=pha(3,3);P(8,8)=std::min(P(8,8)+pha(3,3),PI*PI);}
    void run(const V4&dx,const M4&q,double meas,const V4&host,const M4&Ph,double sigma){t_oi_si=host;M4 f{};M5 T=T5(t_oi_si_prev);for(int r=0;r<4;r++)for(int c=0;c<4;c++)f(r,c)=T(r,c);M4 tmp{};for(int r=0;r<4;r++)for(int c=0;c<4;c++)for(int i=0;i<4;i++)for(int j=0;j<4;j++)tmp(r,c)+=f(r,i)*Ph(i,j)*f(c,j);M4 qh=tmp;predictOdom(dx,q);predictUKF();double R=sigma*sigma;R+=std::sqrt(qh(0,0)*qh(0,0)+qh(1,1)*qh(1,1)+qh(2,2)*qh(2,2));update(meas,R);t_oi_si_prev=t_oi_si;Dt=V4{};qca=M4{};resetDrift(qh);weight*=likelihood;}
};

class UPF{
public: V4 host{};M4 Ph{};vector<UKFParticle> p;double sigma=.01,resample_factor=.1;int na=4,ne=2,nh=4;
    void init(double r){double dalt=PI/3;vector<double> alts{-PI/3,0,PI/3};vector<double> surf;for(double a:alts)surf.push_back((std::sin(a+dalt/2)-std::sin(a-dalt/2))/2);// Use an exact 32-particle grid: 2 + 4 + 2 azimuth bins across the 3 altitude bands, with 4 heading bins.
        // The previous equal-area rounding produced 3 + 4 + 3 azimuth bins = 40 particles while the benchmark was configured/reported as 32.
        int bins[3]={2,4,2};for(int i=0;i<3;i++){double saz=(2*PI/bins[i])/std::sqrt(-8*std::log(.5));double sel=(PI/3)/std::sqrt(-8*std::log(.5));for(int az=0;az<bins[i];az++)for(int h=0;h<nh;h++){V4 s;s[0]=r;s[1]=-PI+(2*PI/bins[i])*az;s[2]=alts[i];s[3]=-PI+(2*PI/nh)*h;UKFParticle q(host,1.0/32.0);q.init(s,2*sigma,saz,sel,s[3],(2*PI/nh)/std::sqrt(-8*std::log(.5)));p.push_back(q);}}}
    void step(const V4&dx,const M4&q,double range,const V4&h){
        host=h; Ph=q;
        if(p.empty()) return;
        double total=0.0;
        for(auto &a:p){ a.run(dx,q,range,host,Ph,sigma); total += std::max(a.weight,1e-300); }
        if(!std::isfinite(total) || total < 1e-300) total=1.0;
        for(auto &a:p) a.weight=std::max(a.weight,1e-300)/total;

        // Safe branch-kill style reduction. Keep at least the strongest particle.
        int best_i=0;
        for(int i=1;i<(int)p.size();++i) if(p[i].weight>p[best_i].weight) best_i=i;
        double threshold=resample_factor/std::max(1,(int)p.size());
        vector<UKFParticle> np;
        for(auto &a:p){
            if(a.weight >= threshold){
                bool merged=false;
                for(auto &k:np) if(close(k,a)){ k.weight += a.weight; merged=true; break; }
                if(!merged) np.push_back(a);
            }
        }
        if(np.empty()) np.push_back(p[best_i]);
        p=np;
        double s=0; for(auto &a:p) s+=a.weight;
        if(s>0) for(auto &a:p) a.weight/=s;
    }
    static bool close(const UKFParticle&a,const UKFParticle&b){V4 A=a.ca_state(),B=b.ca_state();double e=std::sqrt((A[0]-B[0])*(A[0]-B[0])+(A[1]-B[1])*(A[1]-B[1])+(A[2]-B[2])*(A[2]-B[2]));return e<.1&&std::abs(A[3]-B[3])<.1;}
    V4 best()const{ V4 z{}; if(p.empty()) return z; int b=0; for(int i=1;i<(int)p.size();i++) if(p[i].weight>p[b].weight) b=i; return p[b].ca_state(); }
};

static V4 inc(const V4&a,const V4&b){double dx=b[0]-a[0],dy=b[1]-a[1];V4 u;double c=std::cos(a[3]),s=std::sin(a[3]);u[0]=c*dx+s*dy;u[1]=-s*dx+c*dy;u[2]=b[2]-a[2];u[3]=limit_angle(b[3]-a[3]);return u;}

int main(){
    try {
        std::cout << "=== UPF-RPE PC C++ REFERENCE ===" << std::endl;
        std::cout << "Core structure: 5 Algebraic 4-DoF + 5 independent UKF workload slots + 1 Yuri-sir-style UPF" << std::endl;
        std::cout << "UPF initial grid: 32 particles (3 altitude bands: 2/4/2 azimuth bins, 4 heading bins)" << std::endl << std::endl;

        const double dt=0.05;
        const int steps=200;
        const double sigma=0.01;
        V4 h0{}; h0[2]=1.0;
        V4 c0{}; c0[0]=3.5; c0[1]=1.0; c0[2]=2.0; c0[3]=0.3;
        const double d0=std::sqrt(3.5*3.5+1.0+1.0);

        std::vector<AlgebraicMethod4DoF> alg;
        for(int i=0;i<5;i++) alg.emplace_back(d0);

        UPF upf; upf.sigma=sigma; upf.init(d0);
        std::vector<UKFParticle> peers;
        for(int i=0;i<5;i++){
            UKFParticle q(h0,1.0);
            V4 s{}; s[0]=d0; s[1]=0; s[2]=0; s[3]=0;
            q.init(s,2*sigma,0.5,0.5,0,0.5);
            peers.push_back(q);
        }

        std::cout << "Initialization complete." << std::endl;
        std::cout << "UPF particles created: " << upf.p.size() << std::endl;
        std::cout << "Starting " << steps << " cycles..." << std::endl << std::endl;

        V4 ph=h0, pc=c0;
        bool finite=true;
        int valid=0;
        auto t0=std::chrono::high_resolution_clock::now();

        for(int k=0;k<steps;k++){
            double t=(k+1)*dt;
            V4 h{},c{};
            h[0]=2*std::cos(0.3*t); h[1]=2*std::sin(0.3*t); h[2]=1; h[3]=limit_angle(0.3*t);
            c[0]=3.5+0.4*std::sin(0.2*t); c[1]=1+0.2*std::cos(0.2*t); c[2]=2+0.2*std::sin(0.17*t); c[3]=0.3+0.1*std::sin(0.15*t);
            V4 uh=inc(ph,h), uc=inc(pc,c);
            double range=std::sqrt((c[0]-h[0])*(c[0]-h[0])+(c[1]-h[1])*(c[1]-h[1])+(c[2]-h[2])*(c[2]-h[2]));
            M4 q{}; q(0,0)=q(1,1)=q(2,2)=1e-4; q(3,3)=1e-5;

            for(int i=0;i<5;i++){
                alg[i].update(range,uh,uc);
                if(alg[i].valid) valid++;
                peers[i].run(uc,q,range,h,q,sigma);
                if(!finiteV9(peers[i].x) || !finiteM9(peers[i].P)) finite=false;
            }
            upf.step(uc,q,range,h);
            if(upf.p.empty()) finite=false;
            for(const auto &a:upf.p) if(!finiteV9(a.x) || !finiteM9(a.P) || !std::isfinite(a.weight)) finite=false;
            ph=h; pc=c;

            if(k==0 || k==9 || k==49 || k==99 || k==199){
                V4 e=upf.best();
                std::cout << "[step " << std::setw(3) << k+1 << "] UPF best = [ "
                          << std::fixed << std::setprecision(8)
                          << e[0] << " " << e[1] << " " << e[2] << " " << e[3]
                          << " ] | particles=" << upf.p.size() << std::endl;
            }
        }

        auto t1=std::chrono::high_resolution_clock::now();
        double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/steps;
        for(auto &a:alg) if(!finiteV4(a.xr)) finite=false;

        std::cout << std::endl << "--- SUMMARY ---" << std::endl;
        std::cout << "Cycles: " << steps << " @ " << (1.0/dt) << " Hz" << std::endl;
        std::cout << "Algebraic estimators: 5" << std::endl;
        std::cout << "Peer UKFs: 5" << std::endl;
        std::cout << "UPF initial particles: 32" << std::endl;
        std::cout << "UPF final active particles: " << upf.p.size() << std::endl;
        std::cout << "Average cycle time: " << std::setprecision(3) << ms << " ms" << std::endl;
        std::cout << "Algebraic valid-solution updates: " << valid << std::endl;
        std::cout << "All states finite: " << (finite ? "YES" : "NO") << std::endl;
        std::cout << "NOTE: Standalone PC reference; no ROS, logging, or hardware I/O." << std::endl;
        std::cout << "NOTE: C++ least-squares uses normal equations for dependency-free compilation." << std::endl;
        return finite ? 0 : 1;
    } catch(const std::exception &e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 2;
    } catch(...) {
        std::cerr << "FATAL ERROR: unknown exception" << std::endl;
        return 3;
    }
}
