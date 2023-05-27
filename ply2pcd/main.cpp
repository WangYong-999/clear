#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <string>
#include<iostream>

using namespace pcl;
using namespace pcl::io;
using namespace std;


int main (int argc, char** argv)
{
  string plyname[9]={{"/home/ply2pcd/data/000000000-gt-pointcloud.ply"},{"/home/ply2pcd/data/000000003-gt-pointcloud.ply"},{"/home/ply2pcd/data/000000005-gt-pointcloud.ply"},{"/home/ply2pcd/data/000000000-input-pointcloud.ply"},{"/home/ply2pcd/data/000000003-input-pointcloud.ply"},{"/home/ply2pcd/data/000000005-input-pointcloud.ply"},{"/home/ply2pcd/data/000000000-output-pointcloud.ply"},{"/home/ply2pcd/data/000000003-output-pointcloud.ply"},{"/home/ply2pcd/data/000000005-output-pointcloud.ply"}};

  string btname[9]={{"/home/ply2pcd/data/000000000-gt-pointcloud.pcd"},{"/home/ply2pcd/data/000000003-gt-pointcloud.pcd"},{"/home/ply2pcd/data/000000005-gt-pointcloud.pcd"},{"/home/ply2pcd/data/000000000-input-pointcloud.pcd"},{"/home/ply2pcd/data/000000003-input-pointcloud.pcd"},{"/home/ply2pcd/data/000000005-input-pointcloud.pcd"},{"/home/ply2pcd/data/000000000-output-pointcloud.pcd"},{"/home/ply2pcd/data/000000003-output-pointcloud.pcd"},{"/home/ply2pcd/data/000000005-output-pointcloud.pcd"}};


  for(int i=0;i<9;i++)
  {
      pcl::PLYReader reader;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    reader.read<pcl::PointXYZ>(plyname[i], *cloud);
    pcl::io::savePCDFile(btname[i], *cloud );
  }
  return 0;
}
