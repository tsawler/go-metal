package training

import (
	"fmt"
	"math"
	"sync"

	"github.com/tsawler/go-metal/tensor"
)

// Optimizer interface defines the methods that all optimizers must implement
type Optimizer interface {
	Step() error   // Updates model parameters based on gradients
	ZeroGrad()     // Resets gradients to zero for all parameters
	GetLR() float64 // Gets current learning rate
	SetLR(lr float64) // Sets learning rate
}

// SGD implements Stochastic Gradient Descent optimizer
type SGD struct {
	parameters   []*tensor.Tensor
	learningRate float64
	momentum     float64
	weightDecay  float64
	dampening    float64
	nesterov     bool
	velocities   map[*tensor.Tensor]*tensor.Tensor
	mutex        sync.RWMutex
}

// NewSGD creates a new SGD optimizer
func NewSGD(parameters []*tensor.Tensor, lr float64, momentum float64, weightDecay float64, dampening float64, nesterov bool) *SGD {
	sgd := &SGD{
		parameters:   parameters,
		learningRate: lr,
		momentum:     momentum,
		weightDecay:  weightDecay,
		dampening:    dampening,
		nesterov:     nesterov,
		velocities:   make(map[*tensor.Tensor]*tensor.Tensor),
	}
	
	// Initialize velocity tensors for momentum
	if momentum > 0 {
		for _, param := range parameters {
			if param.RequiresGrad() {
				velocity, _ := tensor.Zeros(param.Shape, param.DType, param.Device)
				sgd.velocities[param] = velocity
			}
		}
	}
	
	return sgd
}

// Step performs a single optimization step
func (sgd *SGD) Step() error {
	sgd.mutex.Lock()
	defer sgd.mutex.Unlock()
	
	for _, param := range sgd.parameters {
		if !param.RequiresGrad() || param.Grad() == nil {
			continue
		}
		
		grad := param.Grad()
		
		// Apply weight decay
		if sgd.weightDecay > 0 {
			// grad = grad + weight_decay * param.data
			weightDecayTerm, err := tensor.Mul(param, tensor.FromScalar(sgd.weightDecay, param.DType, param.Device))
			if err != nil {
				return fmt.Errorf("weight decay multiplication failed: %v", err)
			}
			grad, err = tensor.Add(grad, weightDecayTerm)
			if err != nil {
				return fmt.Errorf("weight decay addition failed: %v", err)
			}
		}
		
		// Apply momentum
		if sgd.momentum > 0 {
			velocity := sgd.velocities[param]
			if velocity == nil {
				// Initialize velocity if not found
				v, err := tensor.Zeros(param.Shape, param.DType, param.Device)
				if err != nil {
					return fmt.Errorf("velocity initialization failed: %v", err)
				}
				velocity = v
				sgd.velocities[param] = velocity
			}
			
			// velocity = momentum * velocity + (1 - dampening) * grad
			momentumTerm, err := tensor.Mul(velocity, tensor.FromScalar(sgd.momentum, param.DType, param.Device))
			if err != nil {
				return fmt.Errorf("momentum term calculation failed: %v", err)
			}
			
			gradTerm, err := tensor.Mul(grad, tensor.FromScalar(1.0-sgd.dampening, param.DType, param.Device))
			if err != nil {
				return fmt.Errorf("gradient term calculation failed: %v", err)
			}
			
			newVelocity, err := tensor.Add(momentumTerm, gradTerm)
			if err != nil {
				return fmt.Errorf("velocity update failed: %v", err)
			}
			
			// Update velocity in-place
			err = velocity.SetData(newVelocity.Data)
			if err != nil {
				return fmt.Errorf("velocity data update failed: %v", err)
			}
			
			// Use velocity as gradient for Nesterov momentum
			if sgd.nesterov {
				// grad = grad + momentum * velocity
				nesterovTerm, err := tensor.Mul(newVelocity, tensor.FromScalar(sgd.momentum, param.DType, param.Device))
				if err != nil {
					return fmt.Errorf("nesterov term calculation failed: %v", err)
				}
				grad, err = tensor.Add(grad, nesterovTerm)
				if err != nil {
					return fmt.Errorf("nesterov update failed: %v", err)
				}
			} else {
				grad = newVelocity
			}
		}
		
		// Apply learning rate and update parameters: param.data = param.data - lr * grad
		lrGrad, err := tensor.Mul(grad, tensor.FromScalar(sgd.learningRate, param.DType, param.Device))
		if err != nil {
			return fmt.Errorf("learning rate scaling failed: %v", err)
		}
		
		newData, err := tensor.Sub(param, lrGrad)
		if err != nil {
			return fmt.Errorf("parameter update failed: %v", err)
		}
		
		// Update parameter data in-place
		err = param.SetData(newData.Data)
		if err != nil {
			return fmt.Errorf("parameter data update failed: %v", err)
		}
	}
	
	return nil
}

// ZeroGrad resets gradients to zero for all parameters
func (sgd *SGD) ZeroGrad() {
	tensor.ZeroGrad(sgd.parameters)
}

// GetLR returns the current learning rate
func (sgd *SGD) GetLR() float64 {
	sgd.mutex.RLock()
	defer sgd.mutex.RUnlock()
	return sgd.learningRate
}

// SetLR sets the learning rate
func (sgd *SGD) SetLR(lr float64) {
	sgd.mutex.Lock()
	defer sgd.mutex.Unlock()
	sgd.learningRate = lr
}

// Adam implements the Adam optimizer
type Adam struct {
	parameters  []*tensor.Tensor
	lr          float64
	beta1       float64
	beta2       float64
	eps         float64
	weightDecay float64
	step        int64
	m           map[*tensor.Tensor]*tensor.Tensor // First moment estimates
	v           map[*tensor.Tensor]*tensor.Tensor // Second moment estimates
	mutex       sync.RWMutex
}

// NewAdam creates a new Adam optimizer
func NewAdam(parameters []*tensor.Tensor, lr, beta1, beta2, eps, weightDecay float64) *Adam {
	adam := &Adam{
		parameters:  parameters,
		lr:          lr,
		beta1:       beta1,
		beta2:       beta2,
		eps:         eps,
		weightDecay: weightDecay,
		step:        0,
		m:           make(map[*tensor.Tensor]*tensor.Tensor),
		v:           make(map[*tensor.Tensor]*tensor.Tensor),
	}
	
	// Initialize moment estimates
	for _, param := range parameters {
		if param.RequiresGrad() {
			m, _ := tensor.Zeros(param.Shape, param.DType, param.Device)
			v, _ := tensor.Zeros(param.Shape, param.DType, param.Device)
			adam.m[param] = m
			adam.v[param] = v
		}
	}
	
	return adam
}

// Step performs a single optimization step
func (adam *Adam) Step() error {
	adam.mutex.Lock()
	defer adam.mutex.Unlock()
	
	adam.step++
	
	// Bias correction factors
	bias1 := 1.0 - math.Pow(adam.beta1, float64(adam.step))
	bias2 := 1.0 - math.Pow(adam.beta2, float64(adam.step))
	
	for _, param := range adam.parameters {
		if !param.RequiresGrad() || param.Grad() == nil {
			continue
		}
		
		grad := param.Grad()
		
		// Apply weight decay
		if adam.weightDecay > 0 {
			// grad = grad + weight_decay * param.data
			weightDecayTerm, err := tensor.Mul(param, tensor.FromScalar(adam.weightDecay, param.DType, param.Device))
			if err != nil {
				return fmt.Errorf("weight decay multiplication failed: %v", err)
			}
			grad, err = tensor.Add(grad, weightDecayTerm)
			if err != nil {
				return fmt.Errorf("weight decay addition failed: %v", err)
			}
		}
		
		// Get moment estimates
		m := adam.m[param]
		v := adam.v[param]
		if m == nil || v == nil {
			// Initialize if not found
			mNew, err := tensor.Zeros(param.Shape, param.DType, param.Device)
			if err != nil {
				return fmt.Errorf("first moment initialization failed: %v", err)
			}
			vNew, err := tensor.Zeros(param.Shape, param.DType, param.Device)
			if err != nil {
				return fmt.Errorf("second moment initialization failed: %v", err)
			}
			m = mNew
			v = vNew
			adam.m[param] = m
			adam.v[param] = v
		}
		
		// Update first moment estimate: m = beta1 * m + (1 - beta1) * grad
		beta1Term, err := tensor.Mul(m, tensor.FromScalar(adam.beta1, param.DType, param.Device))
		if err != nil {
			return fmt.Errorf("first moment beta1 term failed: %v", err)
		}
		
		gradTerm, err := tensor.Mul(grad, tensor.FromScalar(1.0-adam.beta1, param.DType, param.Device))
		if err != nil {
			return fmt.Errorf("first moment grad term failed: %v", err)
		}
		
		newM, err := tensor.Add(beta1Term, gradTerm)
		if err != nil {
			return fmt.Errorf("first moment update failed: %v", err)
		}
		
		// Update second moment estimate: v = beta2 * v + (1 - beta2) * grad^2
		beta2Term, err := tensor.Mul(v, tensor.FromScalar(adam.beta2, param.DType, param.Device))
		if err != nil {
			return fmt.Errorf("second moment beta2 term failed: %v", err)
		}
		
		gradSquared, err := tensor.Mul(grad, grad)
		if err != nil {
			return fmt.Errorf("gradient squaring failed: %v", err)
		}
		
		gradSquaredTerm, err := tensor.Mul(gradSquared, tensor.FromScalar(1.0-adam.beta2, param.DType, param.Device))
		if err != nil {
			return fmt.Errorf("second moment grad squared term failed: %v", err)
		}
		
		newV, err := tensor.Add(beta2Term, gradSquaredTerm)
		if err != nil {
			return fmt.Errorf("second moment update failed: %v", err)
		}
		
		// Update moment estimates in-place
		err = m.SetData(newM.Data)
		if err != nil {
			return fmt.Errorf("first moment data update failed: %v", err)
		}
		
		err = v.SetData(newV.Data)
		if err != nil {
			return fmt.Errorf("second moment data update failed: %v", err)
		}
		
		// Bias-corrected estimates
		mHat, err := tensor.Mul(newM, tensor.FromScalar(1.0/bias1, param.DType, param.Device))
		if err != nil {
			return fmt.Errorf("first moment bias correction failed: %v", err)
		}
		
		vHat, err := tensor.Mul(newV, tensor.FromScalar(1.0/bias2, param.DType, param.Device))
		if err != nil {
			return fmt.Errorf("second moment bias correction failed: %v", err)
		}
		
		// Compute update: lr * m_hat / (sqrt(v_hat) + eps)
		vHatSqrt, err := tensor.Sqrt(vHat)
		if err != nil {
			return fmt.Errorf("second moment sqrt failed: %v", err)
		}
		
		denominator, err := tensor.Add(vHatSqrt, tensor.FromScalar(adam.eps, param.DType, param.Device))
		if err != nil {
			return fmt.Errorf("denominator computation failed: %v", err)
		}
		
		update, err := tensor.Div(mHat, denominator)
		if err != nil {
			return fmt.Errorf("update division failed: %v", err)
		}
		
		lrUpdate, err := tensor.Mul(update, tensor.FromScalar(adam.lr, param.DType, param.Device))
		if err != nil {
			return fmt.Errorf("learning rate scaling failed: %v", err)
		}
		
		// Update parameters: param.data = param.data - lr_update
		newData, err := tensor.Sub(param, lrUpdate)
		if err != nil {
			return fmt.Errorf("parameter update failed: %v", err)
		}
		
		// Update parameter data in-place
		err = param.SetData(newData.Data)
		if err != nil {
			return fmt.Errorf("parameter data update failed: %v", err)
		}
	}
	
	return nil
}

// ZeroGrad resets gradients to zero for all parameters
func (adam *Adam) ZeroGrad() {
	tensor.ZeroGrad(adam.parameters)
}

// GetLR returns the current learning rate
func (adam *Adam) GetLR() float64 {
	adam.mutex.RLock()
	defer adam.mutex.RUnlock()
	return adam.lr
}

// SetLR sets the learning rate
func (adam *Adam) SetLR(lr float64) {
	adam.mutex.Lock()
	defer adam.mutex.Unlock()
	adam.lr = lr
}