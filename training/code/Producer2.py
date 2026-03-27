
class setWorkers_mock():
    def __init__(self, indexer, pre_fun, app_fun, buffer_size=100, workers=1):
        self._len = int(len(indexer))
        self._buffer  = int(buffer_size)
        self._workers = int(workers)

        assert(self._workers <=self._buffer)
        assert(3*self._buffer<=self._len   )
        
        self._indexer    = indexer
        self._app_fun    = app_fun
        self._pre_fun    = pre_fun
        self._next_dict  = dict()
        self._next_index = 0
        self._flag_reset = hasattr(self._indexer, "on_epoch_end")
        if self._flag_reset:
            self._indexer.on_epoch_end() 
    
    def info(self):
        print('mock')
        
    def __enter__(self):
        self._next_dict  = dict()
        self._next_index = 0
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
        
    def __len__(self):
        return self._len
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._next_index>= self._len:
            self._next_index = 0
            if self._flag_reset:
                self._indexer.on_epoch_end()
            else:
                raise StopIteration
        
        item = self._indexer[self._next_index]
        item = self._pre_fun(*item)
        item = self._app_fun(*item)
        self._next_index = self._next_index + 1
        #print(self._next_index, end="\r")
        return item

from multiprocessing import Queue   as QueueProcess
from multiprocessing import Process

class setWorkers_process():
    def __init__(self, indexer, pre_fun, app_fun, buffer_size=100, workers=1):
        self._len = int(len(indexer))
        self._buffer  = int(buffer_size)
        self._workers = int(workers)
        self._flag_reset = hasattr(indexer, "on_epoch_end")
        
        assert(self._workers <=self._buffer)
        assert(3*self._buffer<=self._len   )
        
        self._queueIn   = QueueProcess(self._buffer)
        self._processIn = Process(target=self._run_in,args=(indexer, self._queueIn, pre_fun, self._flag_reset ))
        self._queueOut  = QueueProcess(self._buffer)
        self._app_fun   = app_fun
        self._processes = [Process(target=self._run_fun,args=(index,)) for index in range(self._workers)]
        
        self._next_dict = dict()
        self._next_index = 0
        
    
    def _run_in(self, indexer, queue, pre_fun, flag_reset):
        while True:
            if flag_reset:
                indexer.on_epoch_end()                        
            for index in range(len(indexer)):
                try:
                    item = indexer[index]
                    item = pre_fun(*item)
                except:
                    item = None
                #print(index, queue.qsize(), self._queueOut.qsize(), end="\r")
                queue.put((index, item))
   
    def _run_fun(self, index_process):
        while True:
            try:
                index, item = self._queueIn.get()
                item = self._app_fun(*item)
                self._queueOut.put((index, item))
            except:
                pass
    
    def info(self):
        for process in self._processes:
            print(process.is_alive(), end='  ')
        print()
        
    def __enter__(self):
        self._next_dict = dict()
        self._next_index = 0
        for process in self._processes:
            process.start()
        self._processIn.start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        for process in self._processes:
            try:
                process.terminate()
            except:
                pass
        try:
            self._processIn.terminate()
        except:
            pass
        
        self._queueIn.close()
        self._queueOut.close()
        
    def __len__(self):
        return self._len
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._next_index >= self._len:
            self._next_index = 0
            if not self._flag_reset:
                raise StopIteration
        
        numWaiting = 0
        #timeStamp = time()
        while not(self._next_index in self._next_dict.keys()):
            index, item = self._queueOut.get()
            self._next_dict[index] = item
            numWaiting = numWaiting+1
            
        #if numWaiting>1: 
        #    print('\nWaiting %d  (in dict %d) %fsec\n'%(numWaiting, len(self._next_dict.keys()), time()-timeStamp))    
        value = self._next_dict.pop(self._next_index)
        self._next_index = self._next_index+1
        
        return value


from queue import Queue as QueueThread
from threading import Thread

class setWorkers_thread():
    def __init__(self, indexer, pre_fun, app_fun, buffer_size=100, workers=1):
        self._len = int(len(indexer))
        self._buffer = int(buffer_size)
        self._workers = int(workers)
        self._flag_reset = hasattr(indexer, "on_epoch_end")

        assert (self._workers <= self._buffer)
        assert (3 * self._buffer <= self._len)

        self._queueIn = QueueThread(self._buffer)
        self._processIn = Thread(target=self._run_in, args=(indexer, self._queueIn, pre_fun, self._flag_reset))
        self._queueOut = QueueThread(self._buffer)
        self._app_fun = app_fun
        self._processes = [Thread(target=self._run_fun, args=(index,)) for index in range(self._workers)]

        self._next_dict = dict()
        self._next_index = 0

    def _run_in(self, indexer, queue, pre_fun, flag_reset):
        while True:
            if flag_reset:
                indexer.on_epoch_end()
            for index in range(len(indexer)):
                try:
                    item = indexer[index]
                    item = pre_fun(*item)
                except:
                    item = None
                #print(index, queue.qsize(), self._queueOut.qsize(), end="\r")
                queue.put((index, item))

    def _run_fun(self, index_process):
        while True:
            try:
                index, item = self._queueIn.get()
                item = self._app_fun(*item)
                self._queueOut.put((index, item))
            except:
                pass

    def info(self):
        for process in self._processes:
            print(process.is_alive(), end='  ')
        print()

    def __enter__(self):
        self._next_dict = dict()
        self._next_index = 0
        for process in self._processes:
            process.start()
        self._processIn.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for process in self._processes:
            try:
                process.terminate()
            except:
                pass
        try:
            self._processIn.terminate()
        except:
            pass


    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        if self._next_index >= self._len:
            self._next_index = 0
            if not self._flag_reset:
                raise StopIteration

        numWaiting = 0
        # timeStamp = time()
        while not (self._next_index in self._next_dict.keys()):
            index, item = self._queueOut.get()
            self._next_dict[index] = item
            numWaiting = numWaiting + 1

        # if numWaiting>1:
        #    print('\nWaiting %d  (in dict %d) %fsec\n'%(numWaiting, len(self._next_dict.keys()), time()-timeStamp))
        value = self._next_dict.pop(self._next_index)
        self._next_index = self._next_index + 1

        return value