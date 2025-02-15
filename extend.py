import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import networkx as nx

class AdvancedPipeFlowSimulation:
    def __init__(self, initial_velocity=2.0, density=1000):
        """
        Initialize advanced pipe flow simulation with branching capabilities
        
        Args:
            initial_velocity (float): Initial fluid velocity in m/s
            density (float): Fluid density in kg/m³ (default: water)
        """
        self.initial_velocity = initial_velocity
        self.density = density
        self.p0 = 101325  # Reference pressure (Pa)
        
        # Use a graph structure for branching pipes
        self.pipe_graph = nx.DiGraph()
        self.next_node_id = 0
        self.particles = {}
        self.colormap = plt.cm.viridis
        
        # For improved aesthetics
        plt.style.use('dark_background')
    
    def add_pipe_section(self, length, diameter, parent_id=None, angle=0):
        """
        Add a pipe section with specified dimensions and optional branching
        
        Args:
            length (float): Length of pipe section in meters
            diameter (float): Diameter of pipe section in meters
            parent_id (int): ID of parent section to connect to (None for root)
            angle (float): Angle of the pipe section in degrees
        
        Returns:
            int: ID of the newly created pipe section
        """
        node_id = self.next_node_id
        self.next_node_id += 1
        
        # Convert angle to radians
        angle_rad = np.deg2rad(angle)
        
        # Calculate position based on parent (if any)
        if parent_id is None:
            start_x, start_y = 0, 0
            self.root_id = node_id
        else:
            parent = self.pipe_graph.nodes[parent_id]
            parent_end_x = parent['start_x'] + parent['length'] * np.cos(parent['angle'])
            parent_end_y = parent['start_y'] + parent['length'] * np.sin(parent['angle'])
            start_x, start_y = parent_end_x, parent_end_y
        
        # Add node to graph
        self.pipe_graph.add_node(
            node_id,
            length=length,
            diameter=diameter,
            start_x=start_x,
            start_y=start_y,
            angle=angle_rad,
            area=np.pi * (diameter/2)**2
        )
        
        # Connect to parent if specified
        if parent_id is not None:
            self.pipe_graph.add_edge(parent_id, node_id)
        
        return node_id
    
    def calculate_flow_distribution(self):
        """Calculate flow distribution through the pipe network"""
        if not self.pipe_graph:
            return
        
        # Traverse the graph from root to leaves
        for node in nx.topological_sort(self.pipe_graph):
            node_data = self.pipe_graph.nodes[node]
            
            if node == self.root_id:
                # Root node gets the initial velocity
                node_data['velocity'] = self.initial_velocity
                node_data['flow_rate'] = node_data['velocity'] * node_data['area']
            else:
                # Get all incoming edges (should be just one in our case)
                predecessors = list(self.pipe_graph.predecessors(node))
                if not predecessors:
                    continue
                
                parent = predecessors[0]
                parent_data = self.pipe_graph.nodes[parent]
                
                # Get all children of the parent
                children = list(self.pipe_graph.successors(parent))
                if not children:
                    continue
                
                # Calculate how many outgoing branches
                num_branches = len(children)
                
                # For now, we'll assume equal flow distribution among branches
                # More sophisticated models could be implemented here
                node_data['flow_rate'] = parent_data['flow_rate'] / num_branches
                node_data['velocity'] = node_data['flow_rate'] / node_data['area']
    
    def calculate_pressures(self):
        """Calculate pressures using Bernoulli's equation"""
        if not self.pipe_graph:
            return
        
        self.calculate_flow_distribution()
        
        # Set root node pressure
        root_data = self.pipe_graph.nodes[self.root_id]
        root_data['pressure'] = self.p0
        
        # Process nodes in topological order
        for node in nx.topological_sort(self.pipe_graph):
            node_data = self.pipe_graph.nodes[node]
            
            if node == self.root_id:
                continue
                
            # Get parent node
            predecessors = list(self.pipe_graph.predecessors(node))
            if not predecessors:
                continue
                
            parent = predecessors[0]
            parent_data = self.pipe_graph.nodes[parent]
            
            # Calculate pressure using Bernoulli's equation
            delta_p = 0.5 * self.density * (parent_data['velocity']**2 - node_data['velocity']**2)
            node_data['pressure'] = parent_data['pressure'] + delta_p
    
    def initialize_particles(self, n_particles_total=1000):
        """Initialize particles across the entire pipe network"""
        if not self.pipe_graph:
            return
        
        self.calculate_pressures()
        
        # Clear existing particles
        self.particles = {}
        
        # Calculate total pipe length for particle distribution
        total_length = sum(data['length'] for node, data in self.pipe_graph.nodes(data=True))
        
        # Distribute particles among pipes based on their length
        remaining_particles = n_particles_total
        
        for node, data in self.pipe_graph.nodes(data=True):
            # Calculate number of particles for this section
            n_particles = int(remaining_particles * (data['length'] / total_length))
            if n_particles <1:
                n_particles = 1
            remaining_particles -= n_particles
            
            # Initialize random positions along this pipe section
            t_values = np.random.uniform(0, 1, n_particles)  # Position along pipe (0 to 1)
            
            # Calculate actual x, y positions
            x_values = data['start_x'] + t_values * data['length'] * np.cos(data['angle'])
            y_values = data['start_y'] + t_values * data['length'] * np.sin(data['angle'])
            
            # Calculate offset from pipe centerline
            offsets = np.random.uniform(-data['diameter']/2.5, data['diameter']/2.5, n_particles)
            
            # Apply offset perpendicular to pipe direction
            perp_angle = data['angle'] + np.pi/2
            x_values += offsets * np.cos(perp_angle)
            y_values += offsets * np.sin(perp_angle)
            
            self.particles[node] = {
                'x': x_values,
                'y': y_values,
                't': t_values,  # Position along pipe (0 to 1)
                'offsets': offsets,  # Offset from centerline
                'colors': np.random.uniform(0, 1, n_particles)  # For coloring by velocity later
            }
    
    def update_particles(self, dt=0.05):
        """Update all particles based on local velocity"""
        if not self.particles:
            return
            
        for node, data in self.pipe_graph.nodes(data=True):
            if node not in self.particles:
                continue
                
            # Get node data and particles
            node_particles = self.particles[node]
            
            # Update position along pipe (t parameter)
            node_particles['t'] += (data['velocity'] / data['length']) * dt
            
            # Find particles that need to be moved to next pipe section
            moved_indices = np.where(node_particles['t'] >= 1.0)[0]
            if len(moved_indices) > 0:
                # Get successor nodes
                successors = list(self.pipe_graph.successors(node))
                
                if not successors:
                    # End of the line, recycle to a random input pipe
                    recycled_t = node_particles['t'][moved_indices] % 1.0
                    node_particles['t'][moved_indices] = recycled_t
                    
                    # Randomize their offsets
                    node_particles['offsets'][moved_indices] = np.random.uniform(
                        -data['diameter']/2.5, data['diameter']/2.5, len(moved_indices))
                else:
                    # Distribute particles to child pipes
                    for idx in moved_indices:
                        # Choose a random successor
                        next_node = np.random.choice(successors)
                        next_data = self.pipe_graph.nodes[next_node]
                        
                        if next_node not in self.particles:
                            self.particles[next_node] = {
                                'x': np.array([]),
                                'y': np.array([]),
                                't': np.array([]),
                                'offsets': np.array([]),
                                'colors': np.array([])
                            }
                        
                        # Transfer particle to next section
                        t_new = node_particles['t'][idx] % 1.0
                        offset_new = node_particles['offsets'][idx]
                        color_new = node_particles['colors'][idx]
                        
                        self.particles[next_node]['t'] = np.append(self.particles[next_node]['t'], t_new)
                        self.particles[next_node]['offsets'] = np.append(self.particles[next_node]['offsets'], offset_new)
                        self.particles[next_node]['colors'] = np.append(self.particles[next_node]['colors'], color_new)
                    
                    # Remove transferred particles
                    node_particles['t'] = np.delete(node_particles['t'], moved_indices)
                    node_particles['offsets'] = np.delete(node_particles['offsets'], moved_indices)
                    node_particles['colors'] = np.delete(node_particles['colors'], moved_indices)
            
            # Update remaining particles' positions based on t and offset
            if len(node_particles['t']) > 0:
                node_particles['x'] = data['start_x'] + node_particles['t'] * data['length'] * np.cos(data['angle'])
                node_particles['y'] = data['start_y'] + node_particles['t'] * data['length'] * np.sin(data['angle'])
                
                # Apply offset perpendicular to pipe direction
                perp_angle = data['angle'] + np.pi/2
                node_particles['x'] += node_particles['offsets'] * np.cos(perp_angle)
                node_particles['y'] += node_particles['offsets'] * np.sin(perp_angle)
    
    def create_colorful_visualization(self):
        """Create a stylish visualization with branching pipes and colorful elements"""
        self.calculate_pressures()
        
        # Create figure with GridSpec for better layout control
        fig = plt.figure(figsize=(16, 12), facecolor='#1f1f2e')
        gs = GridSpec(3, 2, height_ratios=[3, 1, 0.2])
        
        # Pipe network view subplot
        ax_network = fig.add_subplot(gs[0, :])
        ax_network.set_facecolor('#1f1f2e')
        ax_network.set_title("Pipe Network Flow Visualization", fontsize=16, color='white')
        
        # Velocity graph subplot
        ax_velocity = fig.add_subplot(gs[1, 0])
        ax_velocity.set_facecolor('#1f1f2e')
        ax_velocity.set_title("Velocity Profile", fontsize=12, color='white')
        
        # Pressure graph subplot
        ax_pressure = fig.add_subplot(gs[1, 1])
        ax_pressure.set_facecolor('#1f1f2e')
        ax_pressure.set_title("Pressure Distribution", fontsize=12, color='white')
        
        # Slider axes
        ax_velocity_slider = fig.add_subplot(gs[2, :])
        ax_velocity_slider.set_facecolor('#1f1f2e')
        
        # Determine min/max values for color mapping
        all_velocities = [data['velocity'] for _, data in self.pipe_graph.nodes(data=True)]
        all_pressures = [data['pressure'] for _, data in self.pipe_graph.nodes(data=True)]
        
        if not all_velocities or not all_pressures:
            return fig, None
            
        v_min, v_max = min(all_velocities), max(all_velocities)
        p_min, p_max = min(all_pressures), max(all_pressures)
        
        v_norm = Normalize(vmin=v_min, vmax=v_max)
        p_norm = Normalize(vmin=p_min, vmax=p_max)
        
        # Plot pipe sections with color gradients
        for node, data in self.pipe_graph.nodes(data=True):
            # Calculate end point
            end_x = data['start_x'] + data['length'] * np.cos(data['angle'])
            end_y = data['start_y'] + data['length'] * np.sin(data['angle'])
            
            # Create pipe section with fancy appearance
            pipe_color = self.colormap(v_norm(data['velocity']))
            
            # Calculate corner points for rounded pipe appearance
            angle_perp = data['angle'] + np.pi/2
            half_d = data['diameter'] / 2
            
            # Four corners of pipe rectangle
            c1x = data['start_x'] + half_d * np.cos(angle_perp)
            c1y = data['start_y'] + half_d * np.sin(angle_perp)
            
            c2x = data['start_x'] - half_d * np.cos(angle_perp)
            c2y = data['start_y'] - half_d * np.sin(angle_perp)
            
            c3x = end_x - half_d * np.cos(angle_perp)
            c3y = end_y - half_d * np.sin(angle_perp)
            
            c4x = end_x + half_d * np.cos(angle_perp)
            c4y = end_y + half_d * np.sin(angle_perp)
            
            # Create polygon for pipe
            pipe_polygon = plt.Polygon(
                [[c1x, c1y], [c2x, c2y], [c3x, c3y], [c4x, c4y]],
                closed=True,
                facecolor=pipe_color,
                edgecolor='#5e5e7a',
                alpha=0.8,
                zorder=1
            )
            ax_network.add_patch(pipe_polygon)
            
            # Add flow direction arrow
            mid_x = (data['start_x'] + end_x) / 2
            mid_y = (data['start_y'] + end_y) / 2
            
            arrow_length = data['length'] * 0.2
            dx = arrow_length * np.cos(data['angle'])
            dy = arrow_length * np.sin(data['angle'])
            
            ax_network.arrow(
                mid_x - dx/2, mid_y - dy/2, dx, dy,
                head_width=data['diameter'] * 0.3,
                head_length=data['diameter'] * 0.5,
                fc='white', ec='white', alpha=0.6,
                zorder=2
            )
            
        # Initialize particles if needed
        if not self.particles:
            self.initialize_particles()
        
        # Prepare data for velocity and pressure graphs
        graph_data = []
        for node, data in self.pipe_graph.nodes(data=True):
            # Calculate position for data point (middle of pipe)
            mid_x = data['start_x'] + (data['length']/2) * np.cos(data['angle'])
            mid_y = data['start_y'] + (data['length']/2) * np.sin(data['angle'])
            
            graph_data.append({
                'node': node,
                'position': (mid_x, mid_y),
                'velocity': data['velocity'],
                'pressure': data['pressure'],
                'diameter': data['diameter']
            })
        
        # Sort by distance from origin
        graph_data.sort(key=lambda x: np.sqrt(x['position'][0]**2 + x['position'][1]**2))
        
        # Extract sorted data for plotting
        positions = np.arange(len(graph_data))
        velocities = [item['velocity'] for item in graph_data]
        pressures = [item['pressure']/1000 for item in graph_data]  # Convert to kPa
        diameters = [item['diameter'] for item in graph_data]
        
        # Create colorful graphs
        velocity_bars = ax_velocity.bar(
            positions, velocities, color=[self.colormap(v_norm(v)) for v in velocities],
            alpha=0.7, edgecolor='white', linewidth=0.5
        )
        ax_velocity.set_xlabel('Pipe Section', color='white')
        ax_velocity.set_ylabel('Velocity (m/s)', color='white')
        ax_velocity.tick_params(colors='white')
        
        pressure_bars = ax_pressure.bar(
            positions, pressures, color=[self.colormap(p_norm(p*1000)) for p in pressures],
            alpha=0.7, edgecolor='white', linewidth=0.5
        )
        ax_pressure.set_xlabel('Pipe Section', color='white')
        ax_pressure.set_ylabel('Pressure (kPa)', color='white')
        ax_pressure.tick_params(colors='white')
        
        # Add color bar for velocity
        sm = plt.cm.ScalarMappable(cmap=self.colormap, norm=v_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_velocity, orientation='horizontal', pad=0.2, aspect=40)
        cbar.set_label('Velocity (m/s)', color='white')
        cbar.ax.tick_params(colors='white')
        
        # Add color bar for pressure
        sm2 = plt.cm.ScalarMappable(cmap=self.colormap, norm=p_norm)
        sm2.set_array([])
        cbar2 = plt.colorbar(sm2, ax=ax_pressure, orientation='horizontal', pad=0.2, aspect=40)
        cbar2.set_label('Pressure (kPa)', color='white')
        cbar2.ax.tick_params(colors='white')
        
        # Collect all particles for plotting
        all_x, all_y, all_colors = [], [], []
        
        for node, node_particles in self.particles.items():
            if len(node_particles['x']) > 0:
                all_x.extend(node_particles['x'])
                all_y.extend(node_particles['y'])
                
                # Color particles by local velocity
                node_data = self.pipe_graph.nodes[node]
                normalized_velocity = v_norm(node_data['velocity'])
                all_colors.extend([normalized_velocity] * len(node_particles['x']))
        
        # Create scatter plot for particles
        scatter = ax_network.scatter(
            all_x, all_y,
            c=all_colors,
            cmap=self.colormap,
            s=10,
            alpha=0.8,
            zorder=3
        )
        
        # Set axis limits for network view
        all_coordinates = [(data['start_x'], data['start_y']) for _, data in self.pipe_graph.nodes(data=True)]
        all_coordinates += [(data['start_x'] + data['length'] * np.cos(data['angle']),
                            data['start_y'] + data['length'] * np.sin(data['angle']))
                           for _, data in self.pipe_graph.nodes(data=True)]
        
        x_coords, y_coords = zip(*all_coordinates)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add some padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        ax_network.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
        ax_network.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        
        ax_network.set_aspect('equal')
        ax_network.set_xlabel('X Position (m)', color='white')
        ax_network.set_ylabel('Y Position (m)', color='white')
        ax_network.tick_params(colors='white')
        
        # Create velocity slider
        velocity_slider = Slider(
            ax=ax_velocity_slider,
            label='Initial Velocity (m/s)',
            valmin=0.5,
            valmax=10.0,
            valinit=self.initial_velocity,
            valstep=0.1,
            color='#5e5e7a'
        )
        
        # Update function for parameters
        def update_simulation(val=None):
            self.initial_velocity = velocity_slider.val
            self.calculate_pressures()
            
            # Update pipe colors and data
            for node, data in self.pipe_graph.nodes(data=True):
                for section in ax_network.findobj(plt.Polygon):
                    if section.zorder == 1:  # Our pipes have zorder=1
                        # Update pipe colors based on new velocities
                        section.set_facecolor(self.colormap(v_norm(data['velocity'])))
            
            # Update velocity and pressure data
            velocities = [data['velocity'] for _, data in self.pipe_graph.nodes(data=True)]
            pressures = [data['pressure']/1000 for _, data in self.pipe_graph.nodes(data=True)]
            
            # Update bars
            for i, item in enumerate(graph_data):
                node_data = self.pipe_graph.nodes[item['node']]
                velocity_bars[i].set_height(node_data['velocity'])
                velocity_bars[i].set_color(self.colormap(v_norm(node_data['velocity'])))
                pressure_bars[i].set_height(node_data['pressure']/1000)
                pressure_bars[i].set_color(self.colormap(p_norm(node_data['pressure'])))
            
            # Update y-axis limits for graphs
            ax_velocity.set_ylim(0, max(velocities) * 1.1)
            ax_pressure.set_ylim(min(pressures) * 0.9, max(pressures) * 1.1)
            
            fig.canvas.draw_idle()
        
        # Connect slider to update function
        velocity_slider.on_changed(update_simulation)
        
        plt.tight_layout()
        
        # Animation update function
        def update_animation(frame):
            self.update_particles()
            
            # Collect updated particles
            all_x, all_y, all_colors = [], [], []
            
            for node, node_particles in self.particles.items():
                if len(node_particles['x']) > 0:
                    all_x.extend(node_particles['x'])
                    all_y.extend(node_particles['y'])
                    
                    # Color particles by local velocity
                    node_data = self.pipe_graph.nodes[node]
                    normalized_velocity = v_norm(node_data['velocity'])
                    all_colors.extend([normalized_velocity] * len(node_particles['x']))
            
            if all_x:
                scatter.set_offsets(np.column_stack((all_x, all_y)))
                scatter.set_color(self.colormap(all_colors))
            
            return scatter,
        
        # Create animation
        ani = FuncAnimation(fig, update_animation, frames=100, interval=50, blit=True)
        
        return fig, ani

def create_example_network():
    # Create simulation instance
    sim = AdvancedPipeFlowSimulation(initial_velocity=2.0)
    
    # Create main flow path
    root = sim.add_pipe_section(length=2, diameter=0.3)
    sec1 = sim.add_pipe_section(length=2, diameter=0.3, parent_id=root)
    
    # Add first branch point
    branch1a = sim.add_pipe_section(length=1.5, diameter=0.2, parent_id=sec1, angle=30)
    branch1b = sim.add_pipe_section(length=1.5, diameter=0.2, parent_id=sec1, angle=-30)
    
    # Continue branches
    branch2a = sim.add_pipe_section(length=1, diameter=0.15, parent_id=branch1a)
    branch2b = sim.add_pipe_section(length=1, diameter=0.15, parent_id=branch1b)
    
    # Add second level of branching
    sim.add_pipe_section(length=0.8, diameter=0.1, parent_id=branch2a, angle=15)
    sim.add_pipe_section(length=0.8, diameter=0.1, parent_id=branch2a, angle=-15)
    sim.add_pipe_section(length=0.8, diameter=0.1, parent_id=branch2b, angle=15)
    sim.add_pipe_section(length=0.8, diameter=0.1, parent_id=branch2b, angle=-15)
    
    return sim

def main():
    # Create network and visualize
    sim = create_example_network()
    fig, ani = sim.create_colorful_visualization()
    plt.show()

if __name__ == "__main__":
    main()