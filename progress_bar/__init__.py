import sys, time
import math

class ProgressBar:

    def __init__(self):
        self.time=0
        self.init_time=0
        self.prefix=''
        self.last_print=0

    def set_prefix(self,prefix):
        self.prefix=prefix
        self.time=0
        self.init_time=0
        self.last_print=0

    def log(self, progress):

        estimate_time = None

        if self.init_time==0:
            self.init_time=time.time()
            self.time=self.init_time
            self.last_print=self.init_time
        else:
            if time.time()-self.last_print < 1:
                self.time=time.time()
                return 

            self.time=time.time()
            
            if progress==0:
                return

            self.last_print=self.time
            estimate_time = (self.time - self.init_time)/progress*(1-progress)

            progress=math.floor(progress*100)
            m, s = divmod(estimate_time, 60)
            h, m = divmod(m, 60)
            h=math.floor(h)
            m=math.floor(m)
            s=math.floor(s)
            
            if h==0:
                if m==0:                    
                    print('{} | {}% | Estimate: {:d} sec'.format(self.prefix,progress,s))                    
                else:
                    print('{} | {}% | Estimate: {:d} min {:d} sec'.format(self.prefix,progress,m,s))                    
            else:
                print('{} | {}% | Estimate: {:d} hr {:d} min {:d} sec'.format(self.prefix,progress,h,m,s))
 
if __name__=='__main__':
    bar = ProgressBar()
    bar.set_prefix('test')
    for i in range(10):
        bar.log(i/10)
        time.sleep(1)