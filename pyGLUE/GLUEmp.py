''' Generalised Likelihood Uncertainty Estimation (GLUE) Framework
    Joost Delsman, Deltares, 2011
    Generic framework to conduct GLUE analyses
    Necessary:
    1) model parameters, that:
        - hold the statistical properties of the a priori parameter space
        - hold the statistical properties of the a posteriori parameter space
        - can return a random value based on the a priori parameter space
        - can accept an evaluated parameter + behavioural / non-behavioural
            statement from the evaluator
    2) a model, that:
        - returns a result set based on a parameter set
        - can be called successively to explore the parameter space
        - resides outside GLUE
        - (can even be outside python)
        - function or wrapper must return a dict with parameters and
           their values, recognised by the evaluator
    3) a model evaluator, that:
        - evaluates the model result by a prescribed set of rules
        - assigns behavioural / non-behavioural
    4) a framework that:
        - initializes a monte-carlo sequence
        - for each run:
            - get parameter values
            - run the model
            - evaluate the model
            - store behavioural runs
    '''
import os, sys
import random, types, datetime, multiprocessing, pp
import numpy as np
import copy_reg

'''savestds = os.dup(1), os.dup(2)
f = open(os.devnull, "w")
os.dup2(f.fileno(), sys.__stdout__.fileno())
os.dup2(f.fileno(), sys.__stderr__.fileno())
 
# place code which writes to C stdout, C stderr here
 
os.dup2(savestds[0], 1)
os.dup2(savestds[1], 2)
f.close()
'''


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

class InputError(Exception):
    '''Exception raised for errors in the input.'''
    pass

class GLUEPrmDistribution(object):
    '''Class for creating a GLUE parameter which:
        - has a statistical distribution
        - returns random values from this distribution

        Arguments:
        name         -- name of parameter
        distribution -- either the name of one of the functions from the
                        random module, or a separate python function that
                        returns a random value from a distribution
        funcargs     -- list or dict with function arguments
        '''
    def __init__(self, name=None, distribution='UNIFORM', funcargs=None):
        self.name = name
        self.distribution = distribution
        self.funcargs = funcargs
    def get_value(self):
        ''' Function to return a random value from the a-priori
            parameter space'''
        return self.distribution(**self.funcargs)
    def __call__(self):
        '''calls get_value if parameter is called as a function'''
        return self.get_value()
    def __str__(self):
        if self.name:
            return self.name
        else: return repr(self)

class GLUEParameter(object):
    '''Class for creating a GLUE parameter which:
        - has a statistical distribution
        - returns random values from this distribution
        - and stores posterior distribution
        
        Arguments:
        name         -- name of parameter
        distribution -- either the name of one of the functions from the
                        random module, or a separate python function that
                        returns a random value from a distribution
        funcargs     -- list or dict with function arguments
        '''
    def __init__(self, name=None, distribution='UNIFORM', funcargs=None):
        self.name = name
        self.distribution = distribution
        self.funcargs = funcargs
        self.parameter = GLUEPrmDistribution(self.name,
                                             self.distribution, self.funcargs)
        self.post_exist = False
        self.post_rid = []
        self.post_values = []
        self.post_nruns = 0
        self.post_seed = 0
    def get_value(self):
        ''' Function to return a random value from the a-priori
            parameter space'''
        return self.parameter.get_value()
    def __str__(self):
        if self.name:
            return self.name
        else: return repr(self)

    def init_posterior(self):
        if self.post_exist:
            del(self.post_rid, self.post_values)
            self.post_exist = False
            self.post_rid = []
            self.post_values = []

    def add_posterior(self, result, nruns, seed=0):
        ''' Add BEHAVIOURAL parameterset to posterior array.
            Arguments:
            result -- list of:
                        rid - BEHAVIOURAL monte carlo run id
                        value - parameter value in run id
            nruns -- nr of monte carlo runs
            seed  -- seed used in randomiser
            '''
        if not self.post_exist:
            self.post_exist = True
        if result != []:
            result = zip(*result)
            self.post_rid += list(result[0])
            self.post_values += list(result[1])
        self.post_nruns = nruns
        self.post_seed = seed
                
    def plot_apriori(self, fig=None):
        '''Plots a histogram of a-priori parameter range
            by calling get_value 10000 times'''
        import matplotlib.pyplot as plt

        test = self.get_value()
        if isinstance(test, list):
            nprm = len(test)
        else: nprm = 1
        if self.post_exist:
            n = self.post_nruns
        else: n = 10000
        
        a = np.zeros((n,nprm))
        for i in xrange(self.post_nruns):
            a[i] = self.get_value()
        if a.shape[0] == 1:
            a = a.T

        if fig == None:
            fig = plt.figure()
            fig.suptitle('a-priori (n='+str(len(a[:,0]))+
                         ') parameter distribution '+self.name,
                         fontsize='x-large')
            ncols = 1
            plot = True
        else:
            ncols = 2
            plot = False

        ax = []
        for prm in range(nprm):
            ha = np.histogram(a[:,prm],bins=50)
            ma = np.median(a[:,prm])
            ca = ha[0] / float(len(a[:,prm]))
            w = ha[1][1]-ha[1][0]
            ax.append(fig.add_subplot(nprm, ncols, (prm*ncols)+1))  # numrows, numcols, fignum
            ax[-1].set_title('a-priori distribution')
            ax[-1].bar(ha[1][:-1],ca,w)
            ax[-1].autoscale()
            ax[-1].vlines(ma, *ax[-1].get_ylim(), color='r',  linestyles='solid')

        if plot:
            plt.show()
        else:
            return ax, ha[1]
        
    def plot_posteriori(self):
        '''Plots a histogram of a-priori and
            posteriori parameter ranges
            '''
        import matplotlib.pyplot as plt
        if self.post_exist:
            test = self.get_value()
            if isinstance(test, list):
                nprm = len(test)
            else: nprm = 1
            
            p = np.atleast_2d(self.post_values)
            if p.shape[0] == 1:
                p = p.T

            fig = plt.figure()
            fig.suptitle('a-priori (n='+str(self.post_nruns)+
                         ') and a-posteriori (n='+str(len(p[:,0]))+
                         ') parameter distribution '+self.name,
                         fontsize='x-large')
            ax, bins = self.plot_apriori(fig)
            for prm in range(nprm):
                hp = np.histogram(p[:,prm],bins=bins)
                mp = np.median(p[:,prm])
                cp = hp[0] / float(len(p[:,prm]))
                w = bins[1]-bins[0]
                ax.append(fig.add_subplot(nprm, 2, (prm*2)+2, sharex=ax[-1]))
                ax[-1].set_title('a-posteriori distribution')
                ax[-1].bar(hp[1][:-1],cp,w)
                ax[-1].autoscale()
                ax[-1].vlines(mp, *ax[-1].get_ylim(), color='r', linestyles='solid')
            plt.show()

class GLUEEvaluator(multiprocessing.Process):
#class GLUEEvaluator(object):            
    '''Class to evaluate model output against predefined rules

        Arguments:
        model        -- reference to the model function / modelwrapper-function
        GLUEparams   -- list of GLUEParameter objects
        stdparams    -- dict of standard arguments to the model function
        rules        -- list with functions to evaluate to True / False
        ruleargs     -- list with dicts of arguments to functions
                        (keys rules == keys rulargs)
                        arguments whose values are keys in result dict passed
                        to evaluatefunction are overwritten by the result dict
                        values
        input_queue  -- multiprocessing queue that holds [rid, GLUEparams]
        result_queue -- multiprocessing queue that holds [rid, GLUEparams]
        '''
    def __init__(self, model=None, GLUEparams=None, stdparams=None,
                 rules=None, ruleargs=None, mp=False,
                 input_queue=None, result_queue=None):
        if mp:
            multiprocessing.Process.__init__(self)
        self.model = model
        if isinstance(GLUEparams, list):
            self.GLUEparams = GLUEparams
        else:
            raise TypeError('GLUEparams should be passed as a list')
        if isinstance(stdparams, dict):
            self.stdparams = stdparams
        else: self.stdparams = {}
        if len(rules) != len(ruleargs):
            raise InputError('Rules and Rule arguments do not match')
        self.rulesandargs = zip(rules, ruleargs)
        self.input_queue = input_queue
        self.result_queue = result_queue

    def __str__(self):
        string = 'GLUE Evaluator\n'
        if len(self.rules) > 0:
            string += 30*'-'
            i=1
            for r, ra in self.rulesandargs:
                string += '\nrule '+str(i)+':\t'+r.__name__+'('
                for k,v in ra.iteritems():
                    string += str(k)+'='+str(v)+', '
                string = string[:-2]+')'
                i += 1
            string += '\n'+30*'-'
        else:
            string = string + 'No rules added yet...'
        return string
            
    def evaluate(self, result):
        '''Evaluate result set by rules'''
        evalresult = []
        for func,args in self.rulesandargs:
            argsloc = args.copy()
            for k, v in argsloc.iteritems():
                if v in result:
                    argsloc[k] = result[v]
            evalresult.append(func(**argsloc))
        return False not in evalresult

    def run_and_evaluate(self, rid):
        ''' Runs model and evaluates result,
            returns [rid, prms, self.GLUEeval.evaluate(result)]
            '''
        try:
            print rid
            prms = dict([[prm.name,prm.get_value()] for prm in self.GLUEparams])
            result = self.model(**dict(prms, **self.stdparams))
            return [rid, prms, self.evaluate(result)]
        except:
            return [rid, prms, False]

    def start(self):
        print 'Evaluator %s started.'%self.name
        return multiprocessing.Process.start(self)
    
    def run(self):
        '''Overloaded run function from multiprocessing Process
            Runs the model given the input taken from the input-queue
            Evaluates model result
            Puts result in result-queue
            '''
        while(True):
            # get new input from input queue
            rid = self.input_queue.get()
            if rid is None:
                #print 'Evaluator %s exiting.'%self.name
                break
            else:
                # run and evaluate (returns [rid, prms, evaluate-result]
                #print '%s working on %i'%(self.name,rid)
                result = self.run_and_evaluate(rid)
                # put 
                self.result_queue.put(result)
        return
        
class GLUEFramework(object):
    '''GLUE Framework class
        Controls the GLUE analysis

        Arguments:
        name -- name of the analysis
        model -- reference to the model function / modelwrapper-function
        GLUEparams -- list of GLUEParameter objects
        stdparams -- dict of standard arguments to the model function
        rules -- list with functions to evaluate to True / False to be used by
                    GLUEEvaluator
        ruleargs -- list with dicts of arguments to rule functions
                    (keys rules == keys ruleargs)
        GLUEeval -- reference to GLUEEvaluator object
        nruns -- number of runs in Monte Carlo analysis
        ncpus -- number of processes to spawn in multiprocessing mode
                 defaults to number of cpus
    '''
    def __init__(self,name='GLUE Framework', model=None, GLUEparams=None,
                 stdparams=None, rules=None, ruleargs=None, GLUEeval=None,
                 nruns=1000, ncpus=multiprocessing.cpu_count()):
        self.__name__ = name
        self.model = model
        if isinstance(GLUEparams, list):
            self.GLUEparams = GLUEparams
        else:
            raise TypeError('GLUEparams should be passed as a list')
        if isinstance(stdparams, dict):
            self.stdparams = stdparams
        else: self.stdparams = {}
        self.rules = rules
        self.ruleargs = ruleargs
        self.GLUEeval = GLUEeval
        self.nruns = nruns
        self.nbehavioural = 0
        self.ncpus = ncpus
        self.curr_prm = None
        self.seed = 0
    def __str__(self):
        string = 'GLUE Framework\n'+30*'-'
        if self.model: string = string+'\nUsing model: '+self.model.__name__+'()'
        else: string = string+'\nNo model attached yet'
        if self.GLUEparams != None:
            string = string+'\nParameters: '+', '.join(map(str,self.GLUEparams))
        else: string = string+'\nNo parameters added yet'
        string = string+'\nNr runs: '+str(self.nruns)
        string = string+'\n'+30*'-'
        return string
        
    def add_param(self,param):
        if param.name in self.GLUEparams:
            print 'Parameter '+param.name+' already in parameter list'
        else:
            self.GLUEparams[param.name] = param

    def update_evaluator(self, GLUEeval):
        if isinstance(GLUEeval, GLUEEvaluator):
            if self.GLUEeval:
                del(self.GLUEeval)
            self.GLUEeval = GLUEeval
    def result_to_param(self, result):
        '''Mappable function to get list of behavioural rids and values'''
        try:
            if result[2]:
                return [result[0], result[1][self.curr_prm]]
            else:
                return False
        except:
            print result
            raise
    def run_GLUE(self, verbose=True):
        '''Run GLUE sequence after resetting posterior distributions of
            parameters

            Now straightforward Monte Carlo implementation
            to do:
            - Markov chain
            '''
        if verbose:
            random.seed(self.seed)
            start = datetime.datetime.now()
            print str(self)
            print 'Running',

        # reset posterior distributions of the parameters
        self.nbehavioural = 0
        for gprm in self.GLUEparams:
            gprm.init_posterior()

        # create evaluator if there is none
        if self.GLUEeval == None:
            args = {'model': self.model,
                    'GLUEparams':self.GLUEparams,
                    'stdparams':self.stdparams,
                    'rules': self.rules,
                    'ruleargs':self.ruleargs}
            self.GLUEeval = GLUEEvaluator(**args)
        
        # run monte carlo analysis
        results = []
        for rid in xrange(self.nruns):
            if verbose and rid % int(self.nruns / 22) == 0:
                sys.stdout.write('.')
            results += [self.GLUEeval.run_and_evaluate(rid)]
        for gprm in self.GLUEparams:
            self.curr_prm = gprm.name
            gprm.add_posterior(filter(None,map(self.result_to_param, results)),
                               self.nruns, self.seed)
            
        self.nbehavioural = len(gprm.post_rid)
        
        if verbose:
            print ('\ndone in %s. (%i/%i behavioural)\n'%
                   (str(datetime.datetime.now()-start),self.nbehavioural,
                    self.nruns))
    def run_GLUEmp(self, verbose=True):
        '''Run GLUE sequence in parallel,
            after resetting posterior distributions of
            parameters

            In parallel, use queues to distribute runs among multiple
            GLUEevaluators
            '''
        if verbose:
            random.seed(self.seed)
            start = datetime.datetime.now()
            print str(self)
            print ('Running in parallel with '+
                    str(self.ncpus) +' processes')
            
        # reset posterior distributions of the parameters
        self.nbehavioural = 0
        for gprm in self.GLUEparams:
            gprm.init_posterior()

        # create multiprocessing queues
        input_q = multiprocessing.Queue()
        result_q = multiprocessing.Queue()

        # create multiple GLUEEvaluators
        GLUEparams = [gprm.parameter for gprm in self.GLUEparams]
        args = {'model': self.model,
                'GLUEparams':GLUEparams,
                'stdparams':self.stdparams,
                'rules': self.rules,
                'ruleargs':self.ruleargs,
                'mp':True,
                'input_queue':input_q,
                'result_queue':result_q}
        self.GLUEeval = [GLUEEvaluator(**args) for i in xrange(self.ncpus)]
        for e in self.GLUEeval:
            e.start()

        # submit runs to input_queue
        for rid in xrange(self.nruns):
            input_q.put(rid)
        for e in self.GLUEeval:
            input_q.put(None)

        #for e in self.GLUEeval:
        #    e.join()

        # get results
        num_jobs = self.nruns
        results = []
        while num_jobs:
            if verbose and (self.nruns-num_jobs) % int(self.nruns / 30) == 0:
                sys.stdout.write('.')
            results += [result_q.get()]
            num_jobs -= 1

        for gprm in self.GLUEparams:
            self.curr_prm = gprm.name
            gprm.add_posterior(filter(None,map(self.result_to_param, results)),
                               self.nruns, self.seed)
            
        self.nbehavioural = len(gprm.post_rid)
        
        if verbose:
            print ('\ndone in %s. (%i/%i behavioural)\n'%
                   (str(datetime.datetime.now()-start),self.nbehavioural,
                    self.nruns))
        return

    def run_GLUEpp(self, verbose=True):
        '''Run GLUE sequence in parallel using parallel python,
            after resetting posterior distributions of
            parameters

            In parallel, use queues to distribute runs among multiple
            GLUEevaluators

            GLUE_pp can be run on clusters as well as on SMP
            '''
        if verbose:
            random.seed(self.seed)
            start = datetime.datetime.now()
            print str(self)
            print ('Running in parallel with '+
                    str(self.ncpus) +' processes')
            
        # reset posterior distributions of the parameters
        self.nbehavioural = 0
        for gprm in self.GLUEparams:
            gprm.init_posterior()

        # create job server
        ppservers = ()
        job_server = pp.Server(self.ncpus, ppservers=ppservers, proto=-1)

        # create one base GLUEEvaluator
        GLUEparams = [gprm.parameter for gprm in self.GLUEparams]
        args = {'model': self.model,
                'GLUEparams':GLUEparams,
                'stdparams':self.stdparams,
                'rules': self.rules,
                'ruleargs':self.ruleargs}
        self.GLUEeval = GLUEEvaluator(**args)

        # submit jobs to jobserver
        jobs = []
        for rid in xrange(self.nruns):
            # Submit a job of evaluating a GLUE run
            f = job_server.submit(self.GLUEeval.run_and_evaluate, (rid,),
                                  modules=('emma_glue','np','pandas',))
            jobs.append((rid, f))

        # Retrieve the result of the calculation
        results = []
        for rid, f in jobs:
            #if verbose and rid % int(self.nruns / 30) == 0:
            #    sys.stdout.write('.')
            results.append(f())
        
        for gprm in self.GLUEparams:
            self.curr_prm = gprm.name
            gprm.add_posterior(filter(None,map(self.result_to_param, results)),
                               self.nruns, self.seed)
            
        self.nbehavioural = len(gprm.post_rid)
        
        if verbose:
            print ('\ndone in %s. (%i/%i behavioural)\n'%
                   (str(datetime.datetime.now()-start),self.nbehavioural,
                    self.nruns))
        return
