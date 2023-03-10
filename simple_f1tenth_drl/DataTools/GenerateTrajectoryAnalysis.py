from matplotlib import pyplot as plt
plt.rcParams['pdf.use14corefonts'] = True

import numpy as np
import glob
import os
import math, cmath

import glob
from matplotlib.ticker import PercentFormatter
from matplotlib.collections import LineCollection

from simple_f1tenth_drl.DataTools.MapData import MapData
from simple_f1tenth_drl.PlannerUtils.TrackLine import TrackLine 
from simple_f1tenth_drl.DataTools.plotting_utils import *
from matplotlib.ticker import MultipleLocator

SAVE_PDF = False
# SAVE_PDF = True


def ensure_path_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


class AnalyseTestLapData:
    def __init__(self):
        self.path = None
        self.vehicle_name = None
        self.map_name = None
        self.states = None
        self.actions = None
        self.map_data = None
        self.std_track = None
        self.summary_path = None
        self.lap_n = 0
        
        self.track_progresses = None

    def explore_folder(self, path):
        vehicle_folders = glob.glob(f"{path}*/")
        print(vehicle_folders)
        print(f"{len(vehicle_folders)} folders found")

        set = 1
        for j, folder in enumerate(vehicle_folders):
            print(f"Vehicle folder being opened: {folder}")
                
            self.process_folder(folder)

    def process_folder(self, folder):
        self.path = folder

        self.vehicle_name = self.path.split("/")[-2]
        print(f"Vehicle name: {self.vehicle_name}")
        
        testing_folders = glob.glob(f"{folder}Testing*/")
        for test_folder in testing_folders:
            self.test_folder = test_folder
            test_folder_name = test_folder.split("/")[-2]
            self.map_name = test_folder_name.split("_")[1].lower()
        
            self.map_data = MapData(self.map_name)
            self.std_track = TrackLine(self.map_name, False)
            self.racing_track = TrackLine(self.map_name, True)

        for self.lap_n in range(1):
            if not self.load_lap_data(): break # no more laps
            self.calculate_state_progress()
            
            self.plot_tracking_accuracy()
            self.plot_speed_graph()
            self.plot_slip_graph()
            self.plot_steering_graph()
            self.plot_speed_graph2()
            self.plot_center_line_deviation()

    def load_lap_data(self):
        try:
            data = np.load(self.test_folder + f"Lap_{self.lap_n}_history_{self.vehicle_name}.npy")
        except Exception as e:
            print(e)
            print(f"No data for: " + f"Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
            return 0
        self.states = data[:, :7]
        self.actions = data[:, 7:]
        
        return 1 # to say success
    
    def calculate_state_progress(self):
        progresses = []
        for i in range(len(self.states)):
            p = self.std_track.calculate_progress_percent(self.states[i, 0:2])
            progresses.append(p)
            
        self.track_progresses = np.array(progresses)

    def plot_tracking_accuracy(self):
        pts = self.states[:, 0:2]
        thetas = self.states[:, 4]
        racing_cross_track = []
        racing_heading_error = []
        for i in range(len(pts)):
            track_heading, deviation = self.racing_track.get_cross_track_heading(pts[i])
            racing_cross_track.append(deviation)
            heading_error = sub_angles_complex(track_heading, thetas[i])
            racing_heading_error.append(heading_error)
            
        plt.figure(1, figsize=(10, 5))
        plt.clf()
        plt.plot(self.track_progresses, racing_cross_track)
        
        plt.title("Tracking Accuracy (m)")
        plt.xlabel("Track Progress (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.test_folder}{self.vehicle_name}_cross_tracking_accuracy_{self.lap_n}.svg", bbox_inches='tight', pad_inches=0)
        
        plt.figure(1, figsize=(10, 5))
        plt.clf()
        plt.plot(self.track_progresses, racing_heading_error)
        
        plt.title("Heading Error (rad)")
        plt.xlabel("Track Progress (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.test_folder}{self.vehicle_name}_heading_error_{self.lap_n}.svg", bbox_inches='tight', pad_inches=0)
        
    def plot_center_line_deviation(self):
        pts = self.states[:, 0:2]
        center_line_deviation = []
        for i in range(len(pts)):
            track_heading, deviation = self.std_track.get_cross_track_heading(pts[i])
            center_line_deviation.append(deviation)
            
        plt.figure(1, figsize=(10, 5))
        plt.clf()
        plt.plot(self.track_progresses, center_line_deviation)
        
        plt.title("Centre line Deviation (m)")
        plt.xlabel("Track Progress (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.test_folder}{self.vehicle_name}_centre_line_deviation_{self.lap_n}.svg", bbox_inches='tight', pad_inches=0)
        
    def plot_speed_graph(self):
        plt.figure(1, figsize=(10, 5))
        plt.clf()
        plt.plot(self.track_progresses[:-1], self.actions[:-1, 1], label="Actions", alpha=0.6)
        plt.plot(self.track_progresses[:-1], self.states[:-1, 3], label="State")
        
        raceline_speeds = []
        for i in range(len(self.track_progresses)):
            raceline_speeds.append(self.racing_track.get_raceline_speed(self.states[i, 0:2]))
        plt.plot(self.track_progresses, raceline_speeds, label="Raceline Speed")
    
        plt.legend()
        plt.title("Speed (m/s)")
        plt.xlabel("Track Progress (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.test_folder}{self.vehicle_name}_speed_{self.lap_n}.svg", bbox_inches='tight', pad_inches=0)  
        
    
    def plot_steering_graph(self):
        plt.figure(1, figsize=(10, 5))
        plt.clf()
        plt.plot(self.track_progresses[:-1], self.states[:-1, 2], label="State")
        plt.plot(self.track_progresses[:-1], self.actions[:-1, 0], label="Actions")
    
        plt.legend()
        plt.title("Steering (rad)")
        plt.xlabel("Track Progress (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.test_folder}{self.vehicle_name}_steering_{self.lap_n}.svg", bbox_inches='tight', pad_inches=0)    
        
    def plot_slip_graph(self):
        plt.figure(1, figsize=(10, 5))
        plt.clf()
        plt.plot(self.track_progresses[:-1], self.states[:-1, 6], label="State")
    
        plt.legend()
        plt.title("Slip (rad)")
        plt.xlabel("Track Progress (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.test_folder}{self.vehicle_name}_slip_{self.lap_n}.svg", bbox_inches='tight', pad_inches=0)
    
    def plot_velocity_heat_map(self): 
        save_path  = self.path + "{self.test_folder}"
        
        plt.figure(1)
        plt.clf()
        points = self.states[:, 0:2]
        vs = self.states[:, 3]
        
        self.map_data.plot_map_img()

        xs, ys = self.map_data.pts2rc(points)
        points = np.concatenate([xs[:, None], ys[:, None]], axis=1)
        points = points.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, 8)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(vs)
        lc.set_linewidth(5)
        line = plt.gca().add_collection(lc)
        cbar = plt.colorbar(line,fraction=0.046, pad=0.04, shrink=0.99)
        cbar.ax.tick_params(labelsize=25)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        name = save_path + f"{self.vehicle_name}_velocity_map_{self.lap_n}"
        # std_img_saving(name)
        plt.savefig(name + ".svg", bbox_inches='tight', pad_inches=0)

def sub_angles_complex(a1, a2): 
    real = math.cos(a1) * math.cos(a2) + math.sin(a1) * math.sin(a2)
    im = - math.cos(a1) * math.sin(a2) + math.sin(a1) * math.cos(a2)

    cpx = complex(real, im)
    phase = cmath.phase(cpx)

    return phase

def esp_left_limits():
    plt.xlim(20, 620)
    plt.ylim(50, 520)

def esp_right_limits():
    plt.xlim(900, 1500)
    plt.ylim(50, 520)

def analyse_folder():
    
    TestData = AnalyseTestLapData()
    TestData.explore_folder("Data/")


if __name__ == '__main__':
    analyse_folder()
