package lda.rtm;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import lda.util.LDADoc;
import lda.util.LDAResult;
import lda.LDA;
import lda.LDAConfig;
import lda.LDAParam;
import lda.rtm.util.RTMDocProb;
import lda.rtm.util.RTMFunction;
import structure.TopicModelParameter;
import utility.MathUtil;
import utility.Util;
import cc.mallet.optimize.LimitedMemoryBFGS;

public class RTM extends LDA
{
	public static final int TRAIN_GRAPH=0;
	public static final int TEST_GRAPH=1;
	
	public double eta[];
	
	/**
	 * Edge (d1, d2) with weight w is represented as w=[train|test]EdgeWegihts.get(d1).get(d2)
	 * In this model, w is binary: 1/0 for RTM, 1/-1 for *MedRTM
	 */
	public ArrayList<HashMap<Integer, Integer>> trainEdgeWeights;
	public int numTrainEdges;
	
	public ArrayList<HashMap<Integer, Integer>> testEdgeWeights;
	public int numTestEdges;
	
	public double weight[];
	
	public double PLR;
	
	public void readCorpus(String corpusFileName) throws Exception
	{
		super.readCorpus(corpusFileName);
		for (int doc=0; doc<numDocs; doc++)
		{
			trainEdgeWeights.add(new HashMap<Integer, Integer>());
			testEdgeWeights.add(new HashMap<Integer, Integer>());
		}
	}
	
	public void readGraph(String graphFileName, int graphType) throws Exception
	{
		BufferedReader br=new BufferedReader(new FileReader(graphFileName));
		String line,seg[];
		int u,v;
		while ((line=br.readLine())!=null)
		{
			seg=line.split("\t");
			u=Integer.valueOf(seg[0]);
			v=Integer.valueOf(seg[1]);
			if (corpus.get(u).docLength()==0 || corpus.get(v).docLength()==0) continue;
			
			if (graphType==TRAIN_GRAPH)
			{
				trainEdgeWeights.get(u).put(v, 1);
				numTrainEdges++;
				if (!param.directed)
				{
					trainEdgeWeights.get(v).put(u, 1);
					numTrainEdges++;
				}
			}
			if (graphType==TEST_GRAPH)
			{
				testEdgeWeights.get(u).put(v, 1);
				numTestEdges++;
				if (!param.directed)
				{
					testEdgeWeights.get(v).put(u, 1);
					numTestEdges++;
				}
			}
		}
		if (graphType==TRAIN_GRAPH && param.negEdge)
		{
			sampleNegEdge();
		}
		br.close();
	}
	
	public void sampleNegEdge()
	{
		int numNegEdges=(int)(numTrainEdges*param.negEdgeRatio),u,v;
		for (int i=0; i<numNegEdges; i++)
		{
			u=randoms.nextInt(numDocs);
			v=randoms.nextInt(numDocs);
			while (u==v || corpus.get(u).docLength()==0 || corpus.get(v).docLength()==0 || trainEdgeWeights.get(u).containsKey(v))
			{
				u=randoms.nextInt(numDocs);
				v=randoms.nextInt(numDocs);
			}
			trainEdgeWeights.get(u).put(v, 0);
		}
	}
	
	public void sample(int numIters)
	{
		for (int iteration=1; iteration<=numIters; iteration++)
		{
			for (int doc=0; doc<numDocs; doc++)
			{
				weight=new double[trainEdgeWeights.get(doc).size()];
				sampleDoc(doc);
			}
			
			computeLogLikelihood();
			perplexity=Math.exp(-logLikelihood/numTestWords);
			
			if (type==TRAIN)
			{
				optimize();
			}
			
			if (iteration%param.showPLRInterval==0) computePLR();
			Util.println("<"+iteration+">"+"\tLog-LLD: "+logLikelihood+"\tPPX: "+perplexity+"\tPLR: "+PLR);
		}
		
		if (type==TRAIN)
		{
			for (int topic=0; topic<param.numTopics; topic++)
			{
				Util.println(topWords(topic, 10));
			}
		}
	}
	
	public void sampleDoc(int docIdx)
	{
		int word,oldTopic,newTopic;
		double topicScores[]=new double[param.numTopics];
		LDADoc doc=corpus.get(docIdx);
		
		int i=0;
		for (int d : trainEdgeWeights.get(docIdx).keySet())
		{
			weight[i]=computeWeight(docIdx, d);
			i++;
		}
		
		int interval=getSampleInterval();
		for (int token=0; token<doc.docLength(); token+=interval)
		{
			word=doc.getWord(token);
			oldTopic=doc.getTopicAssign(token);
			if(oldTopic<0 || oldTopic >= param.numTopics)
			    continue;
			if (topics.get(oldTopic).totalTokens==0) continue;
			
			doc.topicCounts[oldTopic]--;
			topics.get(oldTopic).totalTokens--;
			topics.get(oldTopic).vocabCounts[word]--;
			i=0;
			for (int d : trainEdgeWeights.get(docIdx).keySet())
			{
				weight[i]-=eta[oldTopic]/doc.docLength()*
						corpus.get(d).topicCounts[oldTopic]/corpus.get(d).docLength();
				i++;
			}
			
			for (int topic=0; topic<param.numTopics; topic++)
			{
				topicScores[topic]=topicUpdating(docIdx, topic, word);
			}
			
			newTopic=MathUtil.selectDiscrete(topicScores);
            if(newTopic < 0 || newTopic >= param.numTopics)
                continue;
			
			doc.setTopicAssign(token, newTopic);
			doc.topicCounts[newTopic]++;
			topics.get(newTopic).totalTokens++;
			topics.get(newTopic).vocabCounts[word]++;
			i=0;
			for (int d : trainEdgeWeights.get(docIdx).keySet())
			{
				weight[i]+=eta[newTopic]/doc.docLength()*
						corpus.get(d).topicCounts[newTopic]/corpus.get(d).docLength();
				i++;
			}
		}
	}
	
	public double topicUpdating(int doc, int topic, int vocab)
	{
		double score=0.0;
		if (type==TRAIN)
		{
			score=(alpha[topic]+corpus.get(doc).topicCounts[topic])*
					(param.beta+topics.get(topic).vocabCounts[vocab])/
					(param.beta*param.numVocab+topics.get(topic).totalTokens);
		}
		else
		{
			score=(alpha[topic]+corpus.get(doc).topicCounts[topic])*phi[topic][vocab];
		}
		
		int i=0;
		double temp;
		for (int d : trainEdgeWeights.get(doc).keySet())
		{
			temp=MathUtil.sigmoid(weight[i]+eta[topic]/corpus.get(doc).docLength()*
					corpus.get(d).topicCounts[topic]/corpus.get(d).docLength());
			score*=(trainEdgeWeights.get(doc).get(d)>0? temp : 1.0-temp);
			i++;
		}
		return score;
	}
	
	public void optimize()
	{
		RTMFunction optimizable=new RTMFunction(this);
		LimitedMemoryBFGS lbfgs=new LimitedMemoryBFGS(optimizable);
		try
		{
			lbfgs.optimize();
		}
		catch (Exception e)
		{
			return;
		}
		for (int topic=0; topic<param.numTopics; topic++)
		{
			eta[topic]=optimizable.parameters[topic];
		}
	}
	
	public double computeWeight(int doc1, int doc2)
	{
		double weight=0.0;
		for (int topic=0; topic<param.numTopics; topic++)
		{
			weight+=eta[topic]*corpus.get(doc1).topicCounts[topic]/corpus.get(doc1).docLength()*
					corpus.get(doc2).topicCounts[topic]/corpus.get(doc2).docLength();
		}
		return weight;
	}
	
	public double computeEdgeProb(int doc1, int doc2)
	{
		return MathUtil.sigmoid(computeWeight(doc1, doc2));
	}
	
	public void computePLR()
	{
		PLR=0.0;
		ArrayList<RTMDocProb> docProbs=new ArrayList<RTMDocProb>();
		for (int doc=0; doc<numDocs; doc++)
		{
			if (testEdgeWeights.get(doc).size()==0) continue;
			docProbs.clear();
			for (int d=0; d<numDocs; d++)
			{
				if (d==doc) continue;
				docProbs.add(new RTMDocProb(d, computeEdgeProb(doc, d)));
			}
			Collections.sort(docProbs);
			for (int i=0; i<docProbs.size(); i++)
			{
				if (testEdgeWeights.get(doc).containsKey(docProbs.get(i).docNo))
				{
					PLR+=i+1;
				}
			}
		}
		PLR/=(double)numTestEdges;
	}
	
	public void addResults(LDAResult result)
	{
		super.addResults(result);
		result.add(LDAResult.PLR, PLR);
	}
	
	public void writePLR(String plrFileName) throws Exception
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(plrFileName));
		ArrayList<RTMDocProb> docProbs=new ArrayList<RTMDocProb>();
		for (int doc=0; doc<numDocs; doc++)
		{
			if (testEdgeWeights.get(doc).size()==0) continue;
			docProbs.clear();
			for (int d=0; d<numDocs; d++)
			{
				if (d==doc) continue;
				docProbs.add(new RTMDocProb(d, computeEdgeProb(doc, d)));
			}
			Collections.sort(docProbs);
			for (int i=0; i<docProbs.size(); i++)
			{
				if (testEdgeWeights.get(doc).containsKey(docProbs.get(i).docNo))
				{
					bw.write(doc+"\t"+docProbs.get(i).docNo+"\t"+(i+1));
					bw.newLine();
				}
			}
		}
		bw.close();
	}
	
	public void writeUserEmbed(String topicCountFileName, String userIdIdxFileName) throws Exception
	{
        HashMap<Integer, String> Idx_Id_map = new HashMap<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(userIdIdxFileName), "UTF-8"));
        String line;
        while((line = reader.readLine()) != null) {
            if(line.length() < 1)
                continue;
            String[] strs = line.split("\t");
            Idx_Id_map.put(Integer.valueOf(strs[0]), strs[1]);
        }
        reader.close();
        System.out.format("[Info]load %d user idx-id pair from %s\n", Idx_Id_map.size(), userIdIdxFileName);

		BufferedWriter bw=new BufferedWriter(new FileWriter(topicCountFileName));
        bw.write(String.format("%d\t%d\n", numDocs, param.numTopics));
		for (int doc=0; doc<numDocs; doc++)
		{
            if(!Idx_Id_map.containsKey(doc)) {
                System.err.format("[err]doc %d not exist in userIdIdxFile %s.\n", doc, userIdIdxFileName);
                return;
            }
            bw.write(Idx_Id_map.get(doc));
			for (int topic=0; topic<param.numTopics; topic++)
			{
				bw.write("\t" + corpus.get(doc).topicCounts[topic]);
			}
			bw.write("\n");
		}
		bw.close();
	}

    public void writeTopicCounts(String topicCountFileName) throws Exception
    {
        BufferedWriter bw=new BufferedWriter(new FileWriter(topicCountFileName));
        bw.write(String.format("%d\t%d\n", numDocs, param.numTopics));
        for (int doc=0; doc<numDocs; doc++)
        {
            for (int topic=0; topic<param.numTopics; topic++)
            {
                bw.write(corpus.get(doc).topicCounts[topic]+" ");
            }
            bw.write("\n");
        }
        bw.close();
    }
	
	public void getNumTestWords()
	{
		numTestWords=numWords;
	}
	
	public int getStartPos()
	{
		return 0;
	}
	
	public int getSampleSize(int docLength)
	{
		return docLength;
	}
	
	public int getSampleInterval()
	{
		return 1;
	}
	
	public void readModel(String modelFilename) throws Exception
	{
		BufferedReader br=new BufferedReader(new FileReader(modelFilename));
		Util.readMatrix(br, phi);
		Util.readVector(br, alpha);
		Util.readVector(br, eta);
		br.close();
	}
	
	public void writeModel(String modelFileName) throws Exception
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(modelFileName));
		Util.writeMatrix(bw, phi);
		Util.writeVector(bw, alpha);
		Util.writeVector(bw, eta);
		bw.close();
	}
	
	public void initVariables()
	{
		super.initVariables();
		trainEdgeWeights=new ArrayList<HashMap<Integer, Integer>>();
		testEdgeWeights=new ArrayList<HashMap<Integer, Integer>>();
		eta=new double[param.numTopics];
	}
	
	public RTM(LDAParam parameters, int num)
	{
		super(parameters);
		param.numTopics = num;
		for (int topic=0; topic<param.numTopics; topic++)
		{
			eta[topic]=randoms.nextGaussian(0.0, MathUtil.sqr(param.nu));
		}
	}

    public RTM(LDAParam parameters)
    {
        super(parameters);
        for (int topic=0; topic<param.numTopics; topic++)
        {
            eta[topic]=randoms.nextGaussian(0.0, MathUtil.sqr(param.nu));
        }
    }
	
	public RTM(RTM RTMTrain, LDAParam parameters)
	{
		super(RTMTrain, parameters);
		for (int topic=0; topic<param.numTopics; topic++)
		{
			eta[topic]=RTMTrain.eta[topic];
		}
	}
	
	public RTM(String modelFileName, LDAParam parameters) throws Exception
	{
		super(modelFileName, parameters);
	}

	public void setNumTopic(int num){ this.param.numTopics = num;}

	public static void main(String args[]) throws Exception
	{
        TopicModelParameter param = new TopicModelParameter(args);

        String folder = String.format("%s/%s/%s", param.m_prefix, param.m_source, param.m_set);
        String fvFile = String.format("%s/%s/%s_features.txt", param.m_prefix, param.m_source, param.m_source);
        String inputFolder = String.format("%s/RTM", folder);
        String outputFolder = String.format("%s/output/%s/%s/", param.m_prefix, param.m_source, param.m_set);

        String seg[] = Thread.currentThread().getStackTrace()[1].getClassName().split("\\.");
        String modelName = seg[seg.length - 1];
        LDAParam parameters = new LDAParam(fvFile);
        parameters.numTopics = param.m_number_of_topics;
        LDAResult trainResults = new LDAResult();
        LDAResult testResults = new LDAResult();

        double[] perf = new double[param.m_crossV];
        for (int i = 0; i < param.m_crossV; i++) {
            String trainCorpusFileName = String.format("%s/%s_corpus_train_%d.txt", inputFolder, param.m_mode, i);
            String trainLinkFileName = String.format("%s/%s_link_train_%d.txt", inputFolder, param.m_mode, i);
            String userIdIdxFileName = String.format("%s/%s_userId_train_%d.txt", inputFolder, param.m_mode, i);

            String modelFileName = String.format("%s/%d/%s_RTM_model_%d.txt",
                    outputFolder, i, param.m_mode, param.m_number_of_topics);
            String userEmbedFileName = String.format("%s/%d/%s_RTM_userEmbed_%d.txt",
                    outputFolder, i, param.m_mode, param.m_number_of_topics);

            RTM RTMTrain = new RTM(parameters);
            RTMTrain.readCorpus(trainCorpusFileName);
            RTMTrain.readGraph(trainLinkFileName, TRAIN_GRAPH);
            RTMTrain.readGraph(trainLinkFileName, TEST_GRAPH);
            RTMTrain.sample(param.m_emIter);
            RTMTrain.addResults(trainResults);
            if (LDAConfig.SLModel) {
                RTMTrain.writeModel(modelFileName);
            }
            RTMTrain.writeUserEmbed(userEmbedFileName, userIdIdxFileName);

            String testCorpusFileName = String.format("%s/corpus_test_%d.txt", inputFolder, i);
            String testTrainLinkFileName = String.format("%s/link_test_train_%d.txt", inputFolder, i);
            String testTestLinkFileName = String.format("%s/link_test_test_%d.txt", inputFolder, i);
            RTM RTMTest = (LDAConfig.SLModel ?
                    new RTM(modelFileName, parameters) :
                    new RTM(RTMTrain, parameters));
            RTMTest.readCorpus(testCorpusFileName);
            RTMTest.readGraph(testTrainLinkFileName, TRAIN_GRAPH);
            RTMTest.readGraph(testTestLinkFileName, TEST_GRAPH);
            RTMTest.sample(param.m_varMaxIter);
            RTMTest.addResults(testResults);

            trainResults.printResults(modelName + " Test PPX: ", LDAResult.PERPLEXITY);
            testResults.printResults(modelName + " Test PPX: ", LDAResult.PERPLEXITY);
            perf[i] = testResults.getPerplexity();
        }

        double mean = 0;
        double var = 0;
        int invalid = 0;
        for (int i = 0; i < perf.length; i++) {
            if (Double.isNaN(perf[i]) || Double.isInfinite(perf[i])) {
                invalid += 1;
                continue;
            }
            mean += perf[i];
        }
        int valid = perf.length - invalid;
        mean = valid > 0 ? mean / valid : 0;
        for (int i = 0; i < perf.length; i++) {
            if (Double.isNaN(perf[i]) || Double.isInfinite(perf[i])) {
                continue;
            }
            var += (perf[i] - mean) * (perf[i] - mean);
        }
        var = valid > 0 ? Math.sqrt(var / valid) : 0;

        System.out.format("[Stat]Perplexity %.3f+/-%.3f\n", mean, var);

		/*String seg[]=Thread.currentThread().getStackTrace()[1].getClassName().split("\\.");
		String modelName=seg[seg.length-1];
		LDAParam parameters=new LDAParam(LDAConfig.vocabFileName);
		LDAResult trainResults=new LDAResult();
		LDAResult testResults=new LDAResult();
		
		RTM RTMTrain=new RTM(parameters);
		RTMTrain.readCorpus(LDAConfig.trainCorpusFileName);
		RTMTrain.readGraph(LDAConfig.trainLinkFileName, TRAIN_GRAPH);
		RTMTrain.readGraph(LDAConfig.trainLinkFileName, TEST_GRAPH);
		RTMTrain.sample(LDAConfig.numTrainIters);
		RTMTrain.addResults(trainResults);
		if (LDAConfig.SLModel)
		{
			RTMTrain.writeModel(LDAConfig.getModelFileName(modelName));
		}
		
		RTM RTMTest=(LDAConfig.SLModel?
				new RTM(LDAConfig.getModelFileName(modelName), parameters):
				new RTM(RTMTrain, parameters));
		RTMTest.readCorpus(LDAConfig.testCorpusFileName);
		RTMTest.readGraph(LDAConfig.testTrainLinkFileName, TRAIN_GRAPH);
		RTMTest.readGraph(LDAConfig.testTestLinkFileName, TEST_GRAPH);
		RTMTest.sample(LDAConfig.numTestIters);
		RTMTest.addResults(testResults);
		
		trainResults.printResults(modelName+" Test PPX: ", LDAResult.PERPLEXITY);
		testResults.printResults(modelName+" Test PPX: ", LDAResult.PERPLEXITY);*/
	}
}
