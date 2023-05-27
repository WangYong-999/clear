import os

os.system("bin/pcd2octomap data/000000000-gt-pointcloud.pcd data/000000000-gt-pointcloud.bt")
os.system("bin/pcd2octomap data/000000003-gt-pointcloud.pcd data/000000003-gt-pointcloud.bt")
os.system("bin/pcd2octomap data/000000005-gt-pointcloud.pcd data/000000005-gt-pointcloud.bt")

os.system("bin/pcd2octomap data/000000000-input-pointcloud.pcd data/000000000-input-pointcloud.bt")
os.system("bin/pcd2octomap data/000000003-input-pointcloud.pcd data/000000003-input-pointcloud.bt")
os.system("bin/pcd2octomap data/000000005-input-pointcloud.pcd data/000000005-input-pointcloud.bt")

os.system("bin/pcd2octomap data/000000000-output-pointcloud.pcd data/000000000-output-pointcloud.bt")
os.system("bin/pcd2octomap data/000000003-output-pointcloud.pcd data/000000003-output-pointcloud.bt")
os.system("bin/pcd2octomap data/000000005-output-pointcloud.pcd data/000000005-output-pointcloud.bt")

os.system("octovis data/000000000-gt-pointcloud.bt")
os.system("octovis data/000000003-gt-pointcloud.bt")
os.system("octovis data/000000005-gt-pointcloud.bt")
os.system("octovis data/000000000-input-pointcloud.bt")
os.system("octovis data/000000003-input-pointcloud.bt")
os.system("octovis data/000000005-input-pointcloud.bt")
os.system("octovis data/000000000-output-pointcloud.bt")
os.system("octovis data/000000003-output-pointcloud.bt")
os.system("octovis data/000000005-output-pointcloud.bt")
