/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "KeyFrame.h"
#include "Converter.h"

#include "SystemSetting.h"
#include "InitKeyFrame.h"
#include "KeyFrameDatabase.h" //When loading the map, we need to add KeyFrame to KeyFrameDatabase.

#include <set>

#include "Frame.h" // Used for initializing frame.nNextId and mnId

#include <mutex>



namespace ORB_SLAM2
{

class MapPoint;
class KeyFrame;
class SystemSetting;


class Map
{
public:
    Map();

    void AddKeyFrame(KeyFrame* pKF);
    void AddMapPoint(MapPoint* pMP);
    void EraseMapPoint(MapPoint* pMP);
    void EraseKeyFrame(KeyFrame* pKF);
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);
    void Save(const string &filename);//保存地图信息
    
    std::vector<KeyFrame*> GetAllKeyFrames();
    std::vector<MapPoint*> GetAllMapPoints();
    std::vector<MapPoint*> GetReferenceMapPoints();

    long unsigned int MapPointsInMap();
    long unsigned  KeyFramesInMap();

    long unsigned int GetMaxKFid();

    void clear();

    vector<KeyFrame*> mvpKeyFrameOrigins;

    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;
    
    void Load(const string &filename,SystemSetting* mySystemSetting);
    MapPoint* LoadMapPoint(ifstream &f);
    KeyFrame* LoadKeyFrame(ifstream &f,SystemSetting* mySystemSetting);

protected:
    std::set<MapPoint*> mspMapPoints;
    std::set<KeyFrame*> mspKeyFrames;
    
    //保存地图点和关键帧 
    void SaveMapPoint(ofstream &f, MapPoint* mp);
    void SaveKeyFrame(ofstream &f, KeyFrame* kf);
    void GetMapPointsIdx();


    std::vector<MapPoint*> mvpReferenceMapPoints;
    std::map<MapPoint*,unsigned long int> mmpnMapPointsIdx;

    long unsigned int mnMaxKFid;

    std::mutex mMutexMap;
};

} //namespace ORB_SLAM

#endif // MAP_H