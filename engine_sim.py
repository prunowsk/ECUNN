import numpy as np
import pandas as pd


#gonna build an engine simulator to give us our training data
class L15A1_sim:
    def __init__(self):
        #specs of engine and their meanings will be here
        self.displacement = 14997.0 * 1e-6 #displacement in cubic meters
        self.cylinders = 4 #number of cylinders in the engine
        self.stoich_afr = 14.7 #stoichiometric air/fuel ratio is 14.7g air to 1g fuel
        self.air_constant = 287.05 #number of joules required to raise the temp of air by 1 deg C

        #Axis for our tables
        #OEM redline is set to a maximum of 6500 rpm, however this will be a race engine and we will be revving to 7k
        self.rpm_axis = np.array([400,600,800,1000,12001,1400,1500,1700,2000,2300,2500,2700,3000,3300,3500,4000,4500,4800,5000,5500.6000,6300,6500,7000])
        #using mbar as the base to make the math a little easier
        self.map_axis = np.array([120,205,290,403,573,742,855,912,968,1000])

        #base volumetric efficiency table of size (rpm_axis,map_axis)
        #table is on a scale of 0.0-1.0 as a percetage of effieciency
        self.base_ve_table = np.zeros((len(self.rpm_axis), len(self.map_axis)))
        #hondata uses an arbitrary integer for the platform our project will use as a test, so we need a separate fuel table
        self.fuel_table = np.zeros((len(self.rpm_axis), len(self.map_axis)))
        #constant to change the ve table to a fuel table value based on
        self.hondata_fuel_scalar = 1.0286e-8
        for i, rpm in enumerate(self.rpm_axis):
            for j,mbar in enumerate(self.map_axis):
                rpm_factor = 1.0-(((rpm-4800)/6500)**2) * 0.25
                map_factor = 0.7 + (mbar/1000) * 0.25
                #since engines can never be perfectly efficient and to avoid bugs, set lower and upper bounds at 45% and 92% ve
                self.base_ve_table[i,j] = np.clip(rpm_factor*map_factor,0.45,0.92)

                #estimate the amount of air in the engine at the given rpm and pressure
                estimated_air = (101325/(self.air_constant*303.15))*(self.displacement/2)*self.base_ve_table[i,j]
                #mass of fuel is based on the estimated mass of air
                target_fuel = estimated_air/self.stoich_afr
                #hold the arbitrary fuel int into the table
                self.fuel_table = int(target_fuel/self.hondata_fuel_scalar)

    def bilinear_interpolate_grid(self, axis_x, axis_y, grid_matrix, x_val, y_val):
        """Standard spatial blending function reusable for any 2D matrix surface."""
        idx_x = np.clip(np.searchsorted(axis_x, x_val) - 1, 0, len(axis_x) - 2)
        idx_y = np.clip(np.searchsorted(axis_y, y_val) - 1, 0, len(axis_y) - 2)
        
        x0, x1 = axis_x[idx_x], axis_x[idx_x+1]
        y0, y1 = axis_y[idx_y], axis_y[idx_y+1]
        
        v00 = grid_matrix[idx_x, idx_y]
        v10 = grid_matrix[idx_x+1, idx_y]
        v01 = grid_matrix[idx_x, idx_y+1]
        v11 = grid_matrix[idx_x+1, idx_y+1]
        
        x_frac = (x_val - x0) / (x1 - x0) if x1 != x0 else 0
        y_frac = (y_val - y0) / (y1 - y0) if y1 != y0 else 0
        
        return (1 - x_frac) * (1 - y_frac) * v00 + \
               x_frac * (1 - y_frac) * v10 + \
               (1 - x_frac) * y_frac * v01 + \
               x_frac * y_frac * v11
    
    def calculate_mbt_ignition(self,rpm,map_mbar):
        base_advance = 12.0
        rpm_adv = (rpm/7000.0) * 24.0
        map_pull = (map_mbar/1000.0) * 15.0
        return np.clip(base_advance+rpm_adv-map_pull,5.0,42.0)

    def simulate_step(self,rpm,map_mbar,custom_table_raw_val,ign_adv, iat_c=30.0):
        #first estimate air density
        pres_pa=map_mbar * 100.0
        temp_k = iat_c + 273.15
        air_density = pres_pa/(self.air_constant*temp_k)
        #get volumetric efficiency at given rpm and pressure
        true_ve = self.bilinear_interpolate_grid(self.rpm_axis,self.map_axis,self.base_ve_table,rpm,map_mbar)
        #use that given ve value from the table to get the mass of air in the motor
        trapped_air_mass= air_density * (self.displacement/2.0) * true_ve
        #use the scalar value and input fuel table val to estimate fuel mass
        fuel_mass = custom_table_raw_val * self.hondata_fuel_scalar

        #use this to calculate the actual afr and then the lambda val compared to stoich
        actual_afr=trapped_air_mass/max(fuel_mass,1e-6)
        lambda_val = actual_afr/self.stoich_afr

        #calc ignition timing and spark delta
        mbt_spark=self.calculate_mbt_ignition(rpm,map_mbar)
        spark_delta = ign_adv-mbt_spark

        #gather spark efficiency and fuel efficiency at the given point based on lambda and spark_delta
        spark_efficiency = np.clip(1.0-0.0015 * (spark_delta**2),0.0,1.0)
        fuel_efficiency = np.clip(1.0-3.0*((lambda_val-0.89)**2),0.0,1.0)
        #calculate torque
        base_torque = 110.0 * (map_mbar/1000.0) * (true_ve/0.85)
        simulated_torque = base_torque * spark_efficiency * fuel_efficiency

        #knock detection
        knock_threshold = 32.0 - (map_mbar / 1000.0) * 12.0 -(iat_c - 30.0) * 0.2 + (1.0 - lambda_val) * 15.0
        knock_count = 0

        if(ign_adv > knock_threshold):
            severity = ign_adv - knock_threshold
            knock_count = int(np.random.poisson(lam=severity * 1.5))
            simulated_torque *= np.clip(1.0-(knock_count * 0.08),0.2,0.1)

        #fake out sensor noise for realism and to add more randomness to training data
        simulated_torque += np.random.normal(0,0.5)
        actual_afr += np.random.normal(0,0.05)

        return{
            "RPM": int(rpm),
            "MAP_mbar": int(map_mbar),
            "VE_Physical": round(true_ve,3),
            "Hondata_Raw_Fuel": int(custom_table_raw_val),
            "Ignition_Advance_Deg": round(ign_adv,1),
            "Simulated_AFR": round(actual_afr,2),
            "Simulated_Torque_Nm": max(0.0,round(simulated_torque,2)),
            "Knock_Count": knock_count
        }

    def generate_dataset(self,sample=1000):
        data_rows = []
        for i in range(sample):
            rpm = np.random.uniform(800,7000)
            map_mbar = np.random.uniform(120,1000)
            #find our baseline map
            baseline_fuel = self.bilinear_interpolate_grid(self.rpm_axis,self.map_axis,self.fuel_table,rpm,map_mbar)

            #creat a random variance in the test data to train the model
            tested_fuel = baseline_fuel + np.random.uniform(-400,400)

            mbt_guess = self.calculate_mbt_ignition(rpm,map_mbar)
            tested_ignition = np.random.uniform(mbt_guess-10.0,mbt_guess+5.0)

            step_result = self.simulate_step(rpm,map_mbar,tested_fuel,tested_ignition)
            data_rows.append(step_result)

        return pd.DataFrame(data_rows)







    