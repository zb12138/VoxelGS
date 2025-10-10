/*
 * @Author: fuchy@stu.pku.edu.cn
 * @LastEditors: Please set LastEditors
 * @Description:
 */
#include <iostream>
#include <queue>
#include <algorithm>
#include "common/TComPointCloud.h"
#include "common/CommonDef.h"
#include "common/TComOctree.h"
#include "common/HighLevelSyntax.h"
extern "C" {

class PyArray
{
private:

public:
    int* data = nullptr;
    int size[3];
    void init(int I, int J, int K,int val=0);
    PyArray() {};
    ~PyArray();
    int& operator()(int i, int j, int k);
    int& vaule(int i, int j, int k);
    void show(int a=0,int b=1);
};
void PyArray::show(int a , int b)
{
    for (size_t i = a; i < b; i++)
    {
        cout << i << ": \t";
        for (size_t j = 0; j < size[1]; j++)
        {
            for (size_t k = 0; k < size[2]; k++)
            {
                cout << vaule(i, j, k) << ' ';
            }
            cout << " | ";
        }
        cout << endl;
    }
}
void PyArray::init(int I, int J, int K,int val)
{
    size[0] = I; size[1] = J; size[2] = K;
    data = new int[I * J * K]();
    for (size_t i = 0; i < I * J * K; i++){
        data[i] = val;
    }
}

PyArray::~PyArray()
{
    if(data)
        delete data;
}
int& PyArray::operator()(int i, int j, int k)
{
    return data[i * (size[1] * size[2]) + j * size[2] + k];
}
int& PyArray::vaule(int i, int j, int k)
{
    return data[i * (size[1] * size[2]) + j * size[2] + k];
}



class GenOctree {
private:
    TComPointCloud* m_pointCloudOrg;    ///< pointer to input point cloud
    // TComPointCloud* m_pointCloudRecon;  ///< pointer to output point cloud
    HighLevelSyntax* m_hls;             ///< pointer to high-level syntax parameters

public:
    GenOctree(TComPointCloud* pointCloudOrg, HighLevelSyntax* hls) {
        m_pointCloudOrg = pointCloudOrg; m_hls = hls;
    };
    ~GenOctree()
    {
        delete m_pointCloudOrg;
        delete m_hls;
        delete Code;
        delete SeqPos;
        delete SeqAttri;
        delete octree;
    };

    vector<UInt>* Code = new vector<UInt>;// the occupancy sequence
    vector<vector<TComOctreeNode>>* octree = new vector<vector<TComOctreeNode>>;// octree data
    PyArray* SeqPos =new PyArray();// the information of the [occupancy,level,octant,bbox]
    PyArray* SeqAttri= new PyArray();// the attribute in each octant

    Int compressAndEncodeGeometry();
    Void GenKparentSeq(UInt K);

private:
    Void treePartition(const TComOctreeNode& cCurrentNode, const UInt& childNodeSizeLog2,
        vector<UInt>* childPointIdx);

};  ///< END CLASS 

Void GenOctree::treePartition(const TComOctreeNode& cCurrentNode, const UInt& childNodeSizeLog2,
    vector<UInt>* childPointIdx) {
    const auto& pointIdx = cCurrentNode.pointIdx;
    const auto bitMask = 1 << childNodeSizeLog2;

    for (auto idx : pointIdx) {
        const auto& pos = (*m_pointCloudOrg)[idx];
        Int childNodeIdx = (!!(Int(pos[0]) & bitMask)) << 2;
        childNodeIdx |= (!!(Int(pos[1]) & bitMask)) << 1;
        childNodeIdx |= (!!(Int(pos[2]) & bitMask));

        childPointIdx[childNodeIdx].push_back(idx);
    }
}



Void GenOctree::GenKparentSeq(UInt K)
{
    UInt pointNum = m_pointCloudOrg->positions().size();
    UInt LevelNum = octree->size();
    UInt nodeNum = octree->at(LevelNum - 1).back().nodeid;

    PyArray & Seq = *SeqPos;
    Seq.init(nodeNum, K, 6, 0);
    for (size_t i = 0; i < Seq.size[0]; i++)
    {
        for (size_t j = 0; j < Seq.size[1]; j++)
        {
            Seq(i, j, 0) = 255;
        }
    }

    PyArray & Attri = *SeqAttri;
    if (m_pointCloudOrg->hasColors()){
        Attri.init(nodeNum,8,3,0);
    };
    UInt n = 0;
    
    for (size_t L = 0; L < LevelNum; L++)
    {
        for (auto node : (octree->at(L)))
        {
            Seq(n, K - 1, 0) = node.oct;
            Seq(n, K - 1, 1) = L + 1;
            Seq(n, K - 1, 2) = node.octant;
            Seq(n, K - 1, 3) = node.pos[0];
            Seq(n, K - 1, 4) = node.pos[1];
            Seq(n, K - 1, 5) = node.pos[2];
            for (size_t k = 0; k < K - 1; k++) {
                copy(&Seq(node.parent - 1, k + 1, 0), &Seq(node.parent - 1, k + 1, Seq.size[2]), &Seq(n, k, 0));
            }
            if (m_pointCloudOrg->hasColors() && L == LevelNum - 1)
            {
                for (size_t c = 0; c < 8; c++)
                {
                    if (node.childPoint[c].size() > 0)
                    {
                        assert(node.childPoint[c].size() == 1);
                        Attri(n, 7-c, 0) = m_pointCloudOrg->getColors()[node.childPoint[c][0]][0];
                        Attri(n, 7-c, 1) = m_pointCloudOrg->getColors()[node.childPoint[c][0]][1];
                        Attri(n, 7-c, 2) = m_pointCloudOrg->getColors()[node.childPoint[c][0]][2];
                    }
                }
            }
            n += 1;
        }
    }
    assert(n == nodeNum);
}



Int GenOctree::compressAndEncodeGeometry() {
    //vector<int>* Code, vector<vector<TComOctreeNode>>* octree
    //Code 
    const TComPointCloud& pcOrg = *m_pointCloudOrg;
    //TComPointCloud& pcRec = *m_pointCloudRecon;

    queue<TComOctreeNode> fifo;

    TComOctreeNode rootNode;
    rootNode.pos = (UInt)0;
    for (UInt i = 0; i < pcOrg.getNumPoint(); i++)
        rootNode.pointIdx.push_back(i);
    fifo.push(rootNode);

    UInt maxBB = std::max({ 1U, m_hls->sps.geomBoundingBoxSize[0], m_hls->sps.geomBoundingBoxSize[1],
                           m_hls->sps.geomBoundingBoxSize[2] });
    UInt nodeSizeLog2 = ceilLog2(maxBB);
    UInt LevelMax = nodeSizeLog2;
    UInt numNodesInCurrentLevel = 1;
    UInt numNodesInNextLevel = 0;

    UInt numReconPoints = 0;
    UInt nodeid = 1;
    octree->clear();
    octree->resize(LevelMax, vector<TComOctreeNode>());
    Code->clear();
    // Code->reserve(4^nodeSizeLog2);
    for (; !fifo.empty(); fifo.pop()) {
        TComOctreeNode& currentNode = fifo.front();

        if (numNodesInCurrentLevel == 0)  ///< check if all nodes in current level are visited
        {
            nodeSizeLog2--;
            numNodesInCurrentLevel = numNodesInNextLevel;
            numNodesInNextLevel = 0;
        }
        numNodesInCurrentLevel--;

        UInt childSizeLog2 = nodeSizeLog2 - 1;
        vector<UInt> childPointIdx[8];
        treePartition(currentNode, childSizeLog2, childPointIdx);

        UInt occupancyCode = 0;
        for (Int i = 0; i < 8; i++) {
                currentNode.childPoint[i] = childPointIdx[i];
            if (childPointIdx[i].size() > 0)
                occupancyCode |= 1 << i;
        }
        Code->push_back(occupancyCode);
        currentNode.oct = occupancyCode;
        
        for (Int i = 0; i < 8; i++) {
            if (childPointIdx[i].size() == 0)
                continue;

            if (childSizeLog2 == 0)  ///< reaching the leaf nodes
            {
                if (m_hls->sps.geomRemoveDuplicateFlag)  ///< remove duplicate points
                {
                    assert(childPointIdx[i].size() == 1);
                    const auto& idx = childPointIdx[i][0];
                    //pcRec[numReconPoints++] = pcOrg[idx];
                }
                else  ///< keep duplicate points
                {

                }
                continue;
            }

            /// create new child
            fifo.emplace();
            auto& childNode = fifo.back();

            Int x = !!(i & 4);
            Int y = !!(i & 2);
            Int z = !!(i & 1);

            childNode.pos[0] = currentNode.pos[0] + (x << childSizeLog2);
            childNode.pos[1] = currentNode.pos[1] + (y << childSizeLog2);
            childNode.pos[2] = currentNode.pos[2] + (z << childSizeLog2);

            childNode.pointIdx = childPointIdx[i];
            childNode.octant = i + 1;
            childNode.nodeid = ++nodeid;
            childNode.parent = currentNode.nodeid;
            currentNode.childNode[i] = childNode.nodeid;
            numNodesInNextLevel++;
        }
        octree->at(LevelMax - nodeSizeLog2).push_back(currentNode);
    }
    //pcRec.setNumPoint(numReconPoints);
    octree->at(0).at(0).parent = 1;
    return 0;
}
 


    GenOctree* GenOctreeInit(double *data,UInt*rgb,UInt64 numpoint) {
        TComPointCloud* m_pointCloudOrg = new TComPointCloud();
        m_pointCloudOrg->setPoints(data, numpoint);
        if (rgb) {
            m_pointCloudOrg->setRGB(rgb);
        }        
        PC_POS bbMin, bbMax;
        HighLevelSyntax* m_hls=new HighLevelSyntax();
        m_pointCloudOrg->computeBoundingBox(bbMin, bbMax);
        for (Int k = 0; k < 3; k++) {
            m_hls->sps.geomBoundingBoxOrigin[k] = UInt(floor(bbMin[k]));
            auto max_k = bbMax[k];//- m_hls.sps.geomBoundingBoxOrigin[k];
            m_hls->sps.geomBoundingBoxSize[k] = UInt(max_k) + 1;

        }
        GenOctree* genOctreeP = new GenOctree(m_pointCloudOrg, m_hls);
        return genOctreeP;
    }

    vector<UInt> * genOctreeInterface(GenOctree* genOctreeP)
    {
        genOctreeP->compressAndEncodeGeometry();
        return genOctreeP->Code;
    }

    vector<Int> * getChildNodeID(vector<vector<TComOctreeNode>>* Octree,int level)
    {
        vector<Int> * ChildPointID = new vector<Int>;
        if(level==Octree->size()-1 || level<0)
        {
            if(level<0)
            {
                level = Octree->size()+level;
            }
            ChildPointID->reserve(Octree->at(level).size() * 8);
            for (auto node : Octree->at(level))
            {
                for (size_t i = 0; i < 8; i++)
                {
                    if (node.childPoint[i].size() > 0)
                    {
                        ChildPointID->push_back(node.childPoint[i][0]);
                    }
                    else
                    {
                        ChildPointID->push_back(-1);
                    }
                }
            }
        }
        else
        {
            ChildPointID->reserve(Octree->at(level).size() * 8);
            for (auto node : Octree->at(level))
            {
                for (size_t i = 0; i < 8; i++)
                {
                    if (node.childNode[i] > 0)
                    {
                        ChildPointID->push_back(node.childNode[i]);
                    }
                    else
                    {
                        ChildPointID->push_back(-1);
                    }
                }
            }
        }
        return ChildPointID;
    }

    void GenKparentSeqInterface(GenOctree* genOctreeP,int K,int **SeqPos,int **SeqAttri)
    {
        genOctreeP->GenKparentSeq(K);
        *SeqPos = genOctreeP->SeqPos->data;
        *SeqAttri = genOctreeP->SeqAttri->data;
        // return genOctreeP->SeqPos->data;
        // genOctreeP->SeqAttri->show(5500,5501);
    }

    vector<vector<TComOctreeNode>>* getOctree(GenOctree* genOctreeP)
    {
        return genOctreeP->octree;
    }

    
    void delete_vector(GenOctree* v) {        
        delete v;
    }
    int vector_size(vector<vector<TComOctreeNode>>* v) {
        return v->size();
    }
    vector<TComOctreeNode>& vector_get(vector<vector<TComOctreeNode>>* v, int i) {
        return v->at(i);
    }
    void vector_push_back(vector<vector<TComOctreeNode>>* v, vector<TComOctreeNode>& i) {
        v->push_back(i);
    }

    // UInt& child_get(vector<UInt>* v[], int cube, int i) {
    //     return v[cube]->at(i);
    // }


    UInt int_size(vector<UInt>* v) {
        return v->size();
    }

    int int_get(vector<Int>* v, int i) {
        // std::cout<<v->at(i)<<std::endl;
        return v->at(i);
    }

    // vector<TComOctreeNode>* new_Nodes() {
    //     return new vector<TComOctreeNode>;
    // }

    int Nodes_size(vector<TComOctreeNode>* v) {
        return v->size();
    }
    struct node
    {
        UInt nodeid;
        UInt octant;
        UInt parent;
        UInt oct;
        // vector<UInt>* pointIdx;
        vector<UInt>* childPoint[8];
        UInt pos[3];
        UInt childNode[8];
    };
    void delete_Nodes(struct node* v) {
        delete v;
    }
    struct node* Nodes_get(vector<TComOctreeNode>* v, int i,bool leaf) {
        struct node * n= new struct node;
        n->nodeid = v->at(i).nodeid;
        n->octant = v->at(i).octant;
        n->oct = v->at(i).oct;
        n->parent = v->at(i).parent;
        n->pos[0] = v->at(i).pos[0];
        n->pos[1] = v->at(i).pos[1];
        n->pos[2] = v->at(i).pos[2];
        
        {
            for(int c=0;c<=7;c++){
                n->childNode[c] = v->at(i).childNode[c];
                n->childPoint[c] = &(v->at(i).childPoint[c]);
            }
        }
        
        return n;
    }
    void Nodes_push_back(vector<TComOctreeNode>* v, TComOctreeNode& i) {
        v->push_back(i);
    }

}