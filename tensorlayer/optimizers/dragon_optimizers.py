from __future__ import absolute_import, division, print_function
import dragon as dg

__all__ = ['Adadelta', 'Adagrad', 'Adam', 'Admax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD', 'Momentum', 'Lamb', 'LARS']

# Add module aliases


# learning_rate=0.001, rho=0.95, epsilon=1e-07, name='Adadelta'
def Adadelta(**kwargs):
    raise NotImplementedError('Adadelta optimizer function not implemented')


# learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07,name='Adagrad'
def Adagrad(**kwargs):
    raise NotImplementedError('Adagrad optimizer function not implemented')


# learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam'
Adam = dg.optimizers.Adam


# learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adamax'
def Admax(**kwargs):
    raise NotImplementedError('Admax optimizer function not implemented')


# learning_rate=0.001, learning_rate_power=-0.5, initial_accumulator_value=0.1,
# l1_regularization_strength=0.0, l2_regularization_strength=0.0, name='Ftrl',l2_shrinkage_regularization_strength=0.0
def Ftrl(**kwargs):
    raise NotImplementedError('Ftrl optimizer function not implemented')


# learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam',
def Nadam(**kwargs):
    raise NotImplementedError('Nadam optimizer function not implemented')


# learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,name='RMSprop'
RMSprop = dg.optimizers.RMSprop

# learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD'
SGD = dg.optimizers.SGD


# learning_rate, momentum, use_locking=False, name='Momentum', use_nesterov=False
def Momentum(**kwargs):
    raise NotImplementedError('Momentum optimizer function not implemented')


def Lamb(**kwargs):
    raise NotImplementedError('Lamb optimizer function not implemented')


def LARS(**kwargs):
    raise NotImplementedError('LARS optimizer function not implemented')
