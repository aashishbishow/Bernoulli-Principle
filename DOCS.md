# InteractivePipeFlowSimulation - Documentation

"""
Interactive Pipe Flow Simulation

This module implements a real-time, interactive simulation of fluid flow through pipes
with varying cross-sectional areas. It visualizes the effects of the continuity equation
and Bernoulli's principle using particle animation and dynamic graphs.

Key Features:
- Interactive sliders for controlling flow parameters
- Real-time visualization of fluid particles
- Dynamic graphs showing velocity and pressure profiles
- Configurable pipe geometry with multiple sections

Usage:
    sim = InteractivePipeFlowSimulation(initial_velocity=2.0)
    sim.add_pipe_section(length=2, diameter=0.3)
    sim.add_pipe_section(length=1, diameter=0.15)
    fig, ani = sim.create_interactive_visualization()
    plt.show()
"""

class InteractivePipeFlowSimulation:
    """
    Main simulation class for modeling and visualizing pipe flow.
    
    Attributes:
        initial_velocity (float): Starting velocity at inlet (m/s)
        density (float): Fluid density (kg/m³)
        sections (list): Storage for pipe section data
        particles (dict): Particle system for visualization
        p0 (float): Reference pressure (Pa)
    """
    
    def __init__(self, initial_velocity=2.0, density=1000):
        """
        Initialize the simulation with default parameters.
        
        Args:
            initial_velocity (float): Initial fluid velocity in m/s
            density (float): Fluid density in kg/m³
        """
        # [implementation details...]
    
    def add_pipe_section(self, length, diameter, position=None):
        """
        Add a new pipe section to the simulation.
        
        Args:
            length (float): Section length in meters
            diameter (float): Inner diameter in meters
            position (float, optional): Starting x-position. If None, connects to previous section.
        """
        # [implementation details...]
    
    def calculate_velocities(self):
        """
        Calculate fluid velocities in all pipe sections using the continuity equation.
        Updates the 'velocity' field for each section in the sections list.
        """
        # [implementation details...]
    
    def calculate_pressures(self):
        """
        Calculate pressures in all pipe sections using Bernoulli's equation.
        Updates the 'pressure' field for each section in the sections list.
        
        Requires velocities to be calculated first.
        """
        # [implementation details...]
    
    def initialize_particles(self, n_particles=100):
        """
        Create particles for flow visualization.
        
        Args:
            n_particles (int): Number of particles to create
        """
        # [implementation details...]
    
    def update_particles(self, dt=0.05):
        """
        Update particle positions based on local fluid velocity.
        
        Args:
            dt (float): Time step for position updates
        """
        # [implementation details...]
    
    def create_interactive_visualization(self):
        """
        Create an interactive matplotlib figure with animation and sliders.
        
        Returns:
            tuple: (matplotlib figure, animation object)
        """
        # [implementation details...]

The simulation demonstrates these core fluid dynamic principles:

    1.Continuity Equation: As fluid flows through changing pipe diameters, the product of area and velocity remains constant. This is visually demonstrated by faster-moving particles in narrow sections.
    2.Bernoulli's Principle: The inverse relationship between velocity and pressure is shown in the dynamic graphs, where pressure drops in sections with higher velocity.
    3.Conservation of Mass: The simulation maintains constant mass flow rate throughout the pipe system, regardless of geometry changes.
    4.Flow Adaptation: The visualization shows how fluid adapts to changing geometries while maintaining physical constraints.