from StatioSim_IGSS import Model
import matplotlib.pyplot as plt

# Create model instance 
model = Model(pop_total=100, step_limit=1000) 

# Run for multiple steps
for i in range(1000):
    model.step()

# Print analytics
print(model.get_analytics())

# Get agent trails plot
fig = model.get_trails()
plt.show()

# Get animation and save
anim = model.get_ani()
anim.save('agents.gif') 