#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

"""The :mod:`~eap.base` module provides basic structures to build evolutionary
algorithms.
"""

import copy
import operator

from collections import deque
from itertools import izip, repeat, count, imap
        
class Tree(list):
    """ Basic N-ary tree class."""
    class Node(object):
        """ Class representing the node of a Tree.
        
            This class share the basic properties of the Tree, so the Tree's
            methods can use them regardless if the treated object is a Tree or a 
            Node.
        """
        @property
        def height(self):
            """ The height of a Node is always 0."""
            return 0
        
        @property 
        def size(self):
            """ The size of a Node is always 1."""
            return 1
            
        @property
        def root(self):
            """ The root of a node is itself."""
            return self
            
        def _getstate(self):
            """ Convert the node back in its base class. This is specially
                useful when pickling a Tree.
            """
            try:
                base = self.base(self)
            except TypeError:
                base = self.base.__new__(self.base)
            finally :
                try:
                    base.__dict__.update(self.__dict__)
                except AttributeError:
                    pass
                return base

    @classmethod
    def create_node(cls, obj):
        """ Create a node that will be added to the Tree.
        
            A node is run-time defined class that inherits from the object
            and the Node class. This inheritance add functionnalities and  
            attributes that simplifies the task of Tree's methods.
        """
        Node = type("Node", (type(obj), cls.Node), {"base": type(obj)})
        try:
            new_node = Node.__new__(Node)
            new_node.__dict__.update(obj.__dict__)
        except AttributeError:
            new_node = Node(obj)
        return new_node
     
    @classmethod        
    def convert_node(cls, node):
        """ Convert node into the proper object either a Tree or a Node."""
        if isinstance(node, cls.Node):
            return node
        elif isinstance(node, Tree):
            if len(node) == 1:
                return node[0]
            return node
        elif isinstance(node, list):
            if len(node) > 1:
                return Tree(node)
            else:
                return cls.create_node(node[0])
        else:
            return cls.create_node(node)

    def __init__(self, content=None):
        """ Initialize a tree with a list `content`.
        
            The first element of the list is the root of the tree, then the
            following elements are the nodes. A node could be a list, then
            representing a subtree.
        """
        for elem in content:
            self.append(self.convert_node(elem))
    
    def _getstate(self):
        """ Return the state of the Tree
            as a list of arbitrary elements. It is mainly
            used for pickling a Tree object.
        """
        return [elem._getstate() for elem in self] 
    
    def __reduce__(self):
        """ Return the class init, the object's state and the object
            dict in a tuple. The function is used to pickle Tree.
        """
        return (self.__class__, (self._getstate(),), self.__dict__)
    
    def __deepcopy__(self, memo):
        """ Deepcopy a Tree by first converting it back to a list of list.
        
            This deepcopy is faster than the default implementation. From
            quick testing, up to 1.6 times faster, and at least 2 times less
            function calls.
        """
        new = self.__class__(self._getstate())
        new.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return new
        
    def __setitem__(self, key, value):
        """ Set the item at `key` with the corresponding `value`.
        """
        list.__setitem__(self, key, self.convert_node(value))
        
    def __setslice__(self, i, j, value):
        """ Set the slice at `i` to `j` with the corresponding `value`.
        """
        list.__setslice__(self, i, j, self.convert_node(value))
            
    def __str__(self):
        """ Return the tree in its original form, a list, as a string."""
        return list.__repr__(self)
        
    def __repr__(self):
        """ Return the Python code to build a copy of the object."""
        module = self.__module__
        name = self.__class__.__name__
        return "%s.%s(%r)" % (module, name, self._getstate())
   
    @property
    def root(self):
        """Return the root element of the tree."""
        return self[0]

    @property
    def size(self):
        """ Return the number of nodes in the tree."""
        return sum(elem.size for elem in self)

    @property
    def height(self):
        """Return the height of the tree."""
        return max(elem.height for elem in self)+1

    def search_subtree_dfs(self, index):
        """ Search the subtree with the corresponding index based on a depth 
            first search.
        """
        if index == 0:
            return self
        total = 0
        for child in self:
            if total == index:
                return child
            nbr_child = child.size
            if nbr_child + total > index:
                return child.search_subtree_dfs(index-total)
            total += nbr_child

    def set_subtree_dfs(self, index, subtree):
        """ Replace the tree with the corresponding index by subtree based
            on a depth-first search.
        """
        if index == 0:
            try:
                self[:] = subtree
            except TypeError:
                del self[1:]
                self[0] = subtree
            return
    
        total = 0
        for i, child in enumerate(self):
            if total == index:
                self[i] = subtree
                return
            nbr_child = child.size
            if nbr_child + total > index:
                child.set_subtree_dfs(index-total, subtree)
                return
            total += nbr_child

    def search_subtree_bfs(self, index):
        """ Search the subtree with the corresponding index based on a 
            breadth-first search.
        """
        if index == 0:
            return self
        queue = deque(self[1:])
        for i in xrange(index):
            subtree = queue.popleft()
            if isinstance(subtree, Tree):
                queue.extend(subtree[1:])
        return subtree

    def set_subtree_bfs(self, index, subtree):
        """ Replace the subtree with the corresponding index by subtree based
            on a breadth-first search.
        """
        if index == 0:
            try:
                self[:] = subtree
            except TypeError:
                del self[1:]
                self[0] = subtree
            return
                
        queue = deque(izip(repeat(self, len(self[1:])), count(1)))
        for i in xrange(index):
            elem = queue.popleft()
            parent = elem[0]
            child  = elem[1]
            if isinstance(parent[child], Tree):
                tree = parent[child]
                queue.extend(izip(repeat(tree, len(tree[1:])), count(1)))
        parent[child] = subtree

class Fitness(object):
    """The fitness is a measure of quality of a solution.

    Fitnesses may be compared using the ``>``, ``<``, ``>=``, ``<=``, ``==``,
    ``!=``. The comparison of those operators is made
    lexicographically. Maximization and minimization are taken
    care off by a multiplication between the :attr:`weights` and the fitness values.
    The comparison can be made between fitnesses of different size, if the
    fitnesses are equal until the extra elements, the longer fitness will be
    superior to the shorter.

    .. note::
       When comparing fitness values that are minimized, ``a > b`` will return
       :data:`True` if *a* is inferior to *b*.
    """
    
    weights = ()
    """The weights are used in the fitness comparison. They are shared among
    all fitnesses of the same type.
    This member is **not** meant to be manipulated since it may influence how
    fitnesses are compared and may
    result in undesirable effects. However if you wish to manipulate it, in 
    order to make the change effective to all fitnesses of the same type, use
    ``FitnessType.weights = new_weights`` or
    ``self.__class__.weights = new_weights`` or from an individual
    ``ind.fitness.__class__.weights = new_weights``.
    """
    
    wvalues = ()
    """Contains the weighted values of the fitness, the multiplication with the
    weights is made when the values are set via the property :attr:`values`.
    Multiplication is made on setting of the values for efficiency.
    
    Generaly it is unnecessary to manipulate *wvalues* as it is an internal
    attribute of the fitness used in the comparison operators.
    """
    
    def __init__(self, values=()):
        self.values = values
        
    def getValues(self):
        try :
            return tuple(map(operator.div, self.wvalues, self.weights))
        except (AttributeError, TypeError):
            return ()
            
    def setValues(self, values):
        try :
            self.wvalues = tuple(map(operator.mul, values, self.weights))
        except (AttributeError, TypeError):
            self.wvalues = ()
            
    def delValue(self):
        self.wvalues = ()

    values = property(getValues, setValues, delValue,
        ("Fitness values. Use directly ``individual.fitness.values = some_value`` "
         "in order to set the fitness and ``del individual.fitness.values`` "
         "in order to clear (invalidate) the fitness. The (unweighted) fitness "
         "can be directly accessed via ``individual.fitness.values``."))
    
    @property 
    def valid(self):
        """Asses if a fitness is valid or not."""
        return len(self.wvalues) != 0

    def isDominated(self, other):
        """In addition to the comparaison operators that are used to sort
        lexically the fitnesses, this method returns :data:`True` if this
        fitness is dominated by the *other* fitness and :data:`False` otherwise.
        The weights are used to compare minimizing and maximizing fitnesses. If
        there is more fitness values than weights, the las weight get repeated
        until the end of the comparaison.
        """
        not_equal = False
        for self_wvalue, other_wvalue in izip(self.wvalues, other.wvalues):
            if self_wvalue > other_wvalue:
                return False
            elif self_wvalue < other_wvalue:
                not_equal = True
        return not_equal
        
    def __gt__(self, other):
        return not self.__le__(other)
        
    def __ge__(self, other):
        return not self.__lt__(other)

    def __le__(self, other):
        if not other:                   # Protection against yamling
            return False
        return self.wvalues <= other.wvalues

    def __lt__(self, other):
        if not other:                   # Protection against yamling
            return False
        return self.wvalues < other.wvalues

    def __eq__(self, other):
        if not other:                   # Protection against yamling
            return False
        return self.wvalues == other.wvalues
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __deepcopy__(self, memo):
        """Replace the basic deepcopy function with a faster one.
        
        It assumes that the elements in the :attr:`values` tuple are 
        immutable and the fitness does not contain any other object 
        than :attr:`values` and :attr:`weights`.
        """
        return self.__class__(self.values)
    
    def __str__(self):
        """ Return the values of the Fitness object."""
        return str(self.values)
    
    def __repr__(self):
        """ Return the Python code to build a copy of the object."""
        module = self.__module__
        name = self.__class__.__name__
        return "%s.%s(%r)" % (module, name, self.values)
        