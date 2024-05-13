import numpy as np
import networkx as nx
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from windrose import WindroseAxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import torch
from zipfile import ZipFile
import io
import os
import itertools
from torch_geometric.data import Dataset
import math
import warnings
warnings.filterwarnings("ignore")

class GraphFarmsDataset(Dataset):
    """ Dataset comprised of PyWake or HAWC2 farmwide simulations that have been parsed into graphs.
        The dataset should be organized as follows:
        root_path
            ├── layout1.zip
            │    ├── graph1.pt
            │    ├── graph2.pt
            │    ...
            ├── layout2.zip
            ...

        Where each zip file contains the different graphs (inflows) for a given layout. 
        The graphs are stored as PyTorch Geometric Data.
        
        
        args:
        root_path: str, the path to the root directory of the dataset
        rel_wd: bool, whether to include the relative wind direction as an edge feature
    """

    def __init__(self, root_path: str, rel_wd=True):
        super().__init__()
        
        self.__root_path = root_path
        
        # get the list of all the zip files and their contents
        zip_list = [os.path.join(path, name) for path, subdirs, files in os.walk(root_path) for name in files]
        
        # create a list of tuples with the zip file path and the contents
        zip_content_list = []
        for zip_path in zip_list:
            with ZipFile(zip_path, 'r') as zf:
                zip_content_list.append(list(zip([zip_path]*len(zf.namelist()),zf.namelist())))
        
        # store the zip file and its contents in a single matrix
        self.zip_matrix = list(itertools.chain(*zip_content_list))
        
        # get the total number of graphs in the dataset
        self.__num_graphs = len(self.zip_matrix)
        
        # initialize the relative wind direction flag and the stats dicts
        self.rel_wd = rel_wd
        
        # get the dataset stats
        self.input_stats, self.output_stats = self.get_dataset_stats()

    @property
    def num_glob_features(self) -> int:
        r"""Returns the number of global features in the dataset."""
        data = self[0]
        data = data[0] if isinstance(data, tuple) else data
        return data.globals.shape[1]

    @property
    def num_glob_output_features(self) -> int:
        r"""Returns the number of global output features in the dataset."""
        data = self[0]
        data = data[0] if isinstance(data, tuple) else data
        return data.globals_y.shape[1]

    @property
    def num_node_output_features(self) -> int:
        r"""Returns the number of node output features in the dataset."""
        data = self[0]
        data = data[0] if isinstance(data, tuple) else data
        return data.y.shape[1]

    def __len__(self):
        return self.__num_graphs

    def __getitem__(self, idx):
        # read the zip file and select the data to load in by index
        with ZipFile(self.zip_matrix[idx][0], 'r') as zf:
            with zf.open(self.zip_matrix[idx][1]) as item:
                stream = io.BytesIO(item.read())
                data = torch.load(stream)

        # add relative wind direction as an edge feature
        if self.rel_wd:
            edge_rel_wd = math.radians(data.globals[1]) - data.edge_attr[:, 1]
            data.edge_attr = torch.cat((data.edge_attr, edge_rel_wd.unsqueeze(1)), dim=1)

        # make sure all features are float
        data.edge_attr = data.edge_attr.float()
        data.pos = data.pos.float()
        data.globals = data.globals.float().unsqueeze(0)

        return data

    def get_dataset_stats(self):
        input_stats = {'node': {'mean': [], 'std': []}, 'edge': {'mean': [], 'std': []}, 'global': {'mean': [], 'std': []}}
        output_stats = {'node': {'mean': [], 'std': []}, 'edge': {'mean': [], 'std': []}, 'global': {'mean': [], 'std': []}}
        all_x = []
        all_y = []
        all_edge_attr = []
        all_globals = []
        for j in range(len(self)):
            all_x.append(self[j].x)
            all_y.append(self[j].y)
            all_edge_attr.append(self[j].edge_attr)
            all_globals.append(self[j].globals)
        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)
        all_edge_attr = torch.cat(all_edge_attr, dim=0)
        all_globals = torch.cat(all_globals, dim=0)
        
        input_stats['node']['mean'] = all_x.mean(dim=0)
        input_stats['node']['std'] = all_x.std(dim=0)
        input_stats['edge']['mean'] = all_edge_attr.mean(dim=0)
        input_stats['edge']['std'] = all_edge_attr.std(dim=0)
        input_stats['global']['mean'] = all_globals.mean(dim=0)
        input_stats['global']['std'] = all_globals.std(dim=0)
        output_stats['node']['mean'] = all_y.mean(dim=0)
        output_stats['node']['std'] = all_y.std(dim=0)
        
        return input_stats, output_stats
        
    def normalize_sample(self, x, edge_attr, w, y):
        x = (x - self.input_stats['node']['mean']) / self.input_stats['node']['std']
        edge_attr = (edge_attr - self.input_stats['edge']['mean']) / self.input_stats['edge']['std']
        w = (w - self.input_stats['global']['mean']) / self.input_stats['global']['std']
        y = (y - self.output_stats['node']['mean']) / self.output_stats['node']['std']
        return x, edge_attr, w, y
    
    def denormalize_sample(self, y_pred, y):
        y_pred = (y_pred * self.output_stats['node']['std']) + self.output_stats['node']['mean']
        y = (y * self.output_stats['node']['std']) + self.output_stats['node']['mean']
        return y_pred, y


def plot_graph(g: Data, ax=None, highlight:list=None, nx_draw_kwargs:dict=None):
    """ Plots a graph using networkx.
        
        args:
        g: torch_geometric.data.Data, the graph to plot
        ax: matplotlib.axes.Axes, the axis to plot the graph on
        highlight: list, the nodes to highlight
        nx_draw_kwargs: dict, the keyword arguments to pass to the nx.draw function
    """
    if nx_draw_kwargs is None:
        nx_draw_kwargs = {'node_color': '#e38a24', "edgecolors": "black", "node_size": 200, "alpha": 1.0, 'linewidths':1, 'width':0.2}
    
    graph = g
    G = to_networkx(graph, to_undirected=True)
    node_pos_dict = {}
    for i in range(graph.num_nodes):
        node_pos_dict[i] = graph.pos[i, :].tolist()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(G, pos=node_pos_dict, ax=ax, **nx_draw_kwargs)
    if highlight:
        for h in highlight:
            ax.plot(node_pos_dict[h][0], node_pos_dict[h][1], 'o', color='red')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_edgecolor('black')
    return ax


def plot_farm_qty(data, plot_x_or_y='x', var_idx_to_plot=0, highlight_turb_idx=None, show_max =False, ax=None, figsize=(8, 6),windrose=True,
                  title=None, label=False, cmap_label=None, cmap='viridis', cmap_range=None, show_cmap=True, show_graph=False):
    """ Plot a quantity on the farm layout.
    
        args:
        data: torch_geometric.data.Data, the data object
        plot_x_or_y: str, the node features to plot, either 'x' or 'y'
        var_idx_to_plot: int, the index of the variable to plot
        highlight_turb_idx: int, the index of a turbine to highlight if needed
        show_max: bool, whether to highlight the turbine with the maximum value
        ax: matplotlib.axes.Axes, the axis to plot the graph on
        figsize: tuple, the size of the figure
        windrose: bool, whether to plot the windrose in a corner of the plot
        title: str, the title of the plot
        label: bool, whether to label the turbines with their index
        cmap_label: str, the label for the colorbar
        cmap: str, the colormap to use
        cmap_range: tuple, the range of the colormap if needed
        show_cmap: bool, whether to show the colorbar
        show_graph: bool, whether to show the graph of the farm layout
    """
    if plot_x_or_y == 'x':
        y = data.x[:, var_idx_to_plot].numpy()
    else:
        y = data.y[:, var_idx_to_plot].numpy()
    coords = data.pos
    # normalize the coords
    coords[:, 0] = (coords[:, 0] - coords[:, 0].min()) / (coords[:, 0].max() - coords[:, 0].min())
    coords[:, 1] = (coords[:, 1] - coords[:, 1].min()) / (coords[:, 1].max() - coords[:, 1].min())
    
    wf_globals = data.globals
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # cmap = sns.color_palette(cmap, as_cmap=True)
    if show_graph:
        node_pos_dict = {}
        for i in range(data.num_nodes):
            node_pos_dict[i] = data.pos[i, :].tolist()
        G = to_networkx(data, to_undirected=True)
        nx.draw_networkx_edges(G, node_pos_dict, ax=ax, width=2, alpha=0.5)
    
    h1 = ax.scatter(coords[:,0], coords[:,1], c=y, s=150, marker='o', linewidth=1.5, edgecolors='black', cmap=cmap)
    divider = make_axes_locatable(ax)
    width = axes_size.AxesY(ax, aspect=1./20)
    if show_cmap:
        pad = axes_size.Fraction(0.6, width)
        cax = divider.append_axes("left", size=width, pad=pad)
        
        # uncomment to get figure 13 of the paper
        # fig = ax.get_figure()
        # cax = fig.add_axes([0.09, 0.56, 0.009, 0.32]) # [left, bottom, width, height]
        plt.colorbar(h1, cax=cax)
        cax.set_ylabel(cmap_label, fontsize=15)
        cax.yaxis.set_ticks_position('left')
        cax.yaxis.set_label_position('left')
        
        # clb.ax.set_title(cmap_label, fontsize=15)
        if cmap_range is not None:
            h1.set_clim(cmap_range[0], cmap_range[1])
           
    ax.set_title(title, fontsize=20)
    plt.tick_params(left=False, right=False, labelleft=True, labelbottom=False, bottom=False)
    
    if label:
        h_offset = 0.022
        v_offset = 0.022
        for i, txt in enumerate(range(len(coords))):
            ax.text(coords[i, 0]+h_offset, coords[i, 1]+v_offset, str(txt), fontsize=8, ha='left', va='bottom')
    
    if (wf_globals is not None) and (windrose is True):
        ws = wf_globals[0].repeat(6).numpy()
        wd = wf_globals[1].repeat(6).numpy()
        
        # wind rose plot
        axins = inset_axes(ax, width="50%", height="50%", bbox_to_anchor=(0.58, 0.75, 0.4, 0.4), bbox_transform=ax.transAxes,
                   loc='lower right', axes_class=WindroseAxes)
        axins.bar(wd, ws, normed=True, opening=0.8, edgecolor='white', nsector=36, color='#0bb5a7')
        xlabels = ('E', '', 'N', '', 'W', '', 'S', '',)
        axins.set_xticklabels(xlabels)
        axins.grid(linewidth=0.5)
        axins.tick_params(axis='y', which='major')
        axins.set_yticklabels([])
        axins.tick_params(axis='x', which='major', pad=-4, labelsize=9)
        for spine in axins.spines.values():
            spine.set_linewidth(0.1)
        
    plt.tight_layout()
    ax.margins(x=0.05, y=0.1)
    ax.set_aspect('equal')
    
    for spine in ax.spines.values():
            spine.set_linewidth(0.0)
            
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    
    if show_max:
        ax.scatter(coords[np.argmax(y),0], coords[np.argmax(y),1], facecolors='none', s=100, linewidth=3, edgecolors='blue')
    
    if highlight_turb_idx is not None:
        h_plot = ax.scatter(coords[highlight_turb_idx,0], coords[highlight_turb_idx,1], facecolors='none', s=120, linewidth=3, edgecolors='red')
        


def plot_farm_qty_diff(data, y_pred, y, var_idx_to_plot=0, show_max =False, ax=None, figsize=(20, 6), windrose=True, show_graph=False):
    """ Plot the difference between the predicted and true quantity on the farm layout.
    
        args:
        data: torch_geometric.data.Data, the data object
        y_pred: torch.Tensor, the predicted quantity
        y: torch.Tensor, the true quantity
        var_idx_to_plot: int, the index of the variable to plot
        show_max: bool, whether to highlight the turbine with the maximum value
        ax: matplotlib.axes.Axes, the axis to plot the graph on
        figsize: tuple, the size of the figure
        windrose: bool, whether to plot the windrose in a corner of the plot
        show_graph: bool, whether to show the graph of the farm layout
    """
    string_list = ['DEL flap [$kN.m$]', 'DEL edge [$kN.m$]', 'DEL FA [$kN.m$]', 'DEL SS [$kN.m$]',
                'DEL torsion [$kN.m$]']
    
    
    y_diff = y_pred - y
    coords = data.pos
    # normalize the coords
    coords[:, 0] = (coords[:, 0] - coords[:, 0].min()) / (coords[:, 0].max() - coords[:, 0].min())
    coords[:, 1] = (coords[:, 1] - coords[:, 1].min()) / (coords[:, 1].max() - coords[:, 1].min())
    
    wf_globals = data.globals.squeeze()
    if ax is None:
        fig, axs = plt.subplots(figsize=figsize, nrows=1, ncols=3)
    
    titles = ['Predicted', 'True', 'Predicted - True']
    to_plot = [y_pred, y, y_diff]
    
    val_range = [np.min((y_pred[:, var_idx_to_plot].min(), y[:, var_idx_to_plot].min())),
                 np.max((y_pred[:, var_idx_to_plot].max(), y[:, var_idx_to_plot].max()))]
    for k, ax in enumerate(axs):
        if show_graph:
            node_pos_dict = {}
            for i in range(data.num_nodes):
                node_pos_dict[i] = data.pos[i, :].tolist()
            G = to_networkx(data, to_undirected=True)
            nx.draw_networkx_edges(G, node_pos_dict, ax=ax, width=2, alpha=0.5)
        if k == 2:
            h1 = ax.scatter(coords[:,0], coords[:,1], c=to_plot[k][:, var_idx_to_plot].numpy(), s=150, marker='o', linewidth=1.5, edgecolors='black', cmap='coolwarm')
            divider = make_axes_locatable(ax)
            width = axes_size.AxesY(ax, aspect=1./20)
            pad = axes_size.Fraction(0.6, width)
            cax = divider.append_axes("left", size=width, pad=pad)
            plt.colorbar(h1, cax=cax)

        else:
            h1 = ax.scatter(coords[:,0], coords[:,1], c=to_plot[k][:, var_idx_to_plot].numpy(), s=150, marker='o', linewidth=1.5, edgecolors='black', cmap='viridis')
            h1.set_clim(val_range[0], val_range[1])
            divider = make_axes_locatable(ax)
            width = axes_size.AxesY(ax, aspect=1./20)
            pad = axes_size.Fraction(0.6, width)
            cax = divider.append_axes("left", size=width, pad=pad)
            plt.colorbar(h1, cax=cax)
                
        cax.set_ylabel(string_list[var_idx_to_plot], fontsize=12)
        ax.set_title(titles[k], fontsize=20)
        ax.tick_params(left=False, right=False, labelleft=True, labelbottom=False, bottom=False)
        
        ax.set_aspect('equal')
        ax.margins(x=0.05, y=0.1)
        
        for spine in ax.spines.values():
                spine.set_linewidth(0.0)
                
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
            
        if (wf_globals is not None) and (windrose is True):
            ws = wf_globals[0].repeat(6).numpy()
            wd = wf_globals[1].repeat(6).numpy()
            
            # wind rose plot
            axins = inset_axes(ax, width="50%", height="50%", bbox_to_anchor=(0.58, 0.75, 0.4, 0.4), bbox_transform=ax.transAxes,
                    loc='lower right', axes_class=WindroseAxes)
            axins.bar(wd, ws, normed=True, opening=0.8, edgecolor='white', nsector=36, color='#0bb5a7')
            xlabels = ('E', '', 'N', '', 'W', '', 'S', '',)
            axins.set_xticklabels(xlabels)
            axins.grid(linewidth=0.5)
            axins.tick_params(axis='y', which='major')
            axins.set_yticklabels([])
            axins.tick_params(axis='x', which='major', pad=-4, labelsize=9)
            for spine in axins.spines.values():
                spine.set_linewidth(0.1)
            
        if show_max:
            ax.scatter(coords[np.argmax(y_diff),0], coords[np.argmax(y_diff),1], facecolors='none', s=100, linewidth=3, edgecolors='blue')
    plt.tight_layout()