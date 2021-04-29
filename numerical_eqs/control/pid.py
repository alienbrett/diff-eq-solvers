
class PIDController:
    '''Shamelessly stolen from pseudocode on wikipedia
    https://www.wikiwand.com/en/PID_controller#/Pseudocode
    '''
    def __init__( self, tunings, setpoint ):
        self.tunings = tuple(tunings)
        self.setpoint = setpoint
        
        self.e = 0
        self.i = 0
    
    def update(self, dt, xt):
        kp, ki, kd = self.tunings
        error = self.setpoint - xt
        proportional = error
        self.i += error * dt
        deriv = ( error - self.e ) / dt
        output = kp * proportional + ki * self.i + kd * deriv
        self.e = error
        
        return output