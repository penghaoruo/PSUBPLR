import java.io.*;
import java.util.*;
import java.net.URI;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.JobContext;

// Stochastic primal update in parallel (for map process) 
class PrimalMap extends Mapper<LongWritable, Text, Text, Text> {
	
	// Number of Iterations
	public int T;
	
	// Number of feature dimension
	public int d;
	
	// Number of data instances
	public int n;
	
	// Learning variables 
	public double w[];
	
	// Bias
	public double b;
	
	// Probability Vector
	public double p[];
	
	// Represent sparse x[] and indicate non-zero indexes
	Vector<Integer> x_index;
		
	// Represent sparse x[] and indicate correponding non-zero values
	Vector<Double> x_value;
	
	// Logistic function
	public double funcg(double tmp) {
		double res=0;
		res=1.0/(double)(1+Math.exp(tmp));
		return res;
	}
	
	// Vector inner multiplication
	public double funcm() {
		double res=0;
		for (int i=0;i<x_index.size();i++)
			res=res+x_value.elementAt(i)*w[i];
		return res;
	}
	
	// Load parameters: w[] and p[]
	public void read_through_hdfs() throws Exception {
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		
		// Read cached hdfs file "paraw"
		Path inFile = new Path("sublinear/tmp/paraw");
		FSDataInputStream in = fs.open(inFile);
		for (int i=0;i<d;i++)
			w[i]=in.readDouble();
		in.close();
		
		// Read cached hdfs file "parap"
		inFile = new Path("sublinear/tmp/parap");
		in = fs.open(inFile);
		for (int i=0;i<n;i++)
			p[i]=in.readDouble();
		in.close();
	}
	
	// Implemented Once
	protected void setup(Mapper.Context context) throws IOException, InterruptedException {
		
		// Load parameters through configuration
		T=context.getConfiguration().getInt("T",0);
		d=context.getConfiguration().getInt("d",0);
		n=context.getConfiguration().getInt("n",0);
		b=context.getConfiguration().getFloat("b",0);
		
		// Initialize w[] and p[]
		w=new double[d];
		p=new double[n];
		
		// Try to load parameters w[] and p[] through cached hdfs file
		x_value=new Vector<Double>();
		x_index=new Vector<Integer>();
		try {
			read_through_hdfs();
		}
		catch(Exception e) {
			Integer.parseInt("3.6");
		}
	}
	
	// Primal map function
	protected void map(LongWritable key, Text value, Context context) throws IOException,InterruptedException {
		
		// Data instance id for p[] 
		int index;
		
		// Data label
		int y;
		
		// Computed gradient value
		double coef=0;
		
		String line = value.toString();
		StringTokenizer itr = new StringTokenizer(line);
		
		// Parse data instance id
		index=Integer.parseInt(itr.nextToken());
		
		// Parse data instance by sparse representation
		String tmp=itr.nextToken();
		while (tmp.contains(":")) {
			String[] strs=tmp.split(":");
			x_index.addElement(Integer.parseInt(strs[0]));
			x_value.addElement(Double.parseDouble(strs[1]));
			tmp=itr.nextToken();
		}
		
		// Parse data label
		y=Integer.parseInt(itr.nextToken());

		// Random Choose Process
		Random rnd=new Random();
		int r=0; //r=1;//r=rnd.nextInt();
		if (p[index-1]>(double)(r)/(double)(n))
			coef=y*funcg(y*(funcm()+b));
		
		// Change sampling process to exception computation
		coef=coef*p[index-1];
		
		// Set key-value pair for reduce
		for (int i=0;i<x_index.size();i++) {
			Text keytext = new Text();
			keytext.set(x_index.elementAt(i).toString());
			
			double temp=coef*x_value.elementAt(i)/Math.sqrt(2*T);
			Text valuetext = new Text();
			valuetext.set((new Double(temp)).toString());
			context.write(keytext, valuetext);
		}
		
		// clear data for function re-call
		x_value.clear();
		x_index.clear();
	}
	
}

// Stochastic primal update in parallel (for reduce process) 
class PrimalReduce extends Reducer<Text, Text, Text, Text> {
	
	// Primal reduce function
	protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException,InterruptedException {
	    double sum = 0;
		while (values.iterator().hasNext())
			sum+=Double.parseDouble(values.iterator().next().toString());
		context.write(key, new Text((new Double(sum)).toString()));
	}
	
}

// Stochastic dual update in parallel (for map process) 
class DualMap extends Mapper<LongWritable, Text, IntWritable, Text> {
	
	// Number of data instances
	public int n;
	
	// Number of feature dimension
	public int d;
	
	// Learning variables 
	public double w[];
		
	// Bias
	public double b;	
	
	// Id for Chosen feature
	public int jt;

	// Input eta value
	public double eta;
	
	// Soft margin vector
	public int kexi[];
	
	// Define clip function
	public double clip(double a, double b) {
		return Math.max(Math.min(a,b),(-1)*b);
	}
	
	// Load parameters: w[] and kexi[]
	public void read_through_hdfs() throws Exception {
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		
		// Read cached hdfs file "paraw"
		Path inFile = new Path("sublinear/tmp/paraw");
		FSDataInputStream in = fs.open(inFile);
		for (int i=0;i<d;i++)
			w[i]=in.readDouble();
		in.close();
		
		// Read cached hdfs file "parakexi"
		inFile = new Path("sublinear/tmp/parakexi");
		in = fs.open(inFile);
		for (int i=0;i<n;i++) 
			kexi[i]=in.readInt();
		in.close();
	}
	
	// Implemented Once
	protected void setup(Mapper.Context context) throws IOException, InterruptedException {
		
		// Load parameters through configuration
		d=context.getConfiguration().getInt("d",0);
		n=context.getConfiguration().getInt("n",0);
		jt=context.getConfiguration().getInt("jt",0);
		b=context.getConfiguration().getFloat("b",0);
		eta=context.getConfiguration().getFloat("eta",0);
		
		// Initialize w[]
		w=new double[d];
		kexi=new int[n];
		
		// Try to load parameters w[] through cached hdfs file
		try {
			read_through_hdfs();
		}
		catch(Exception e) {
			Integer.parseInt("3.6");
		}
	}
	
	// Dual map function
	protected void map(LongWritable key, Text value, Context context) throws IOException,InterruptedException {
		
		// Data instance id for p[] 
		int index;
				
		// Data label
		int y;
		
		// Chosen feature value - x[jt]
		double x_value=0;
		
		// 2-norm value of x[]
		double len;
		
		String line = value.toString();
		StringTokenizer itr = new StringTokenizer(line);
		
		// Parse data instance id
		index=Integer.parseInt(itr.nextToken());
		
		// Parse data instance to get the chosen feature value
		String tmp=itr.nextToken();
		while (tmp.contains(":")) {
			int i_index=Integer.parseInt(tmp.split(":")[0]);
			
			// note: svmlight ensures features are in order
			if (i_index>jt) break;
			if (i_index<jt) {
				tmp=itr.nextToken();
				continue;
			}
			if (i_index==jt) {
				x_value=Double.parseDouble(tmp.split(":")[1]);
				break;
			}
		}
		while (tmp.contains(":")) tmp=itr.nextToken();
		
		// Parse 2-norm value of x[]
		len=Double.parseDouble(tmp);
		
		// Parse data label
		y=Integer.parseInt(itr.nextToken());
		
		// Multiplicative weights update method
		double sigma=x_value*len*len/w[jt]+kexi[index-1]+b*y;
		double sigma_hat=clip(sigma,1.0/eta);
		double res=1-eta*sigma_hat+eta*sigma_hat*eta*sigma_hat;
		
		// Set key-value pair for reduce
		context.write(new IntWritable(index-1), new Text((new Double(res)).toString()));
	}
	
}

// Main Class for Sublinear Logistic Regression with l2-penalty
public class SublinearLogisticLtwo {
	
	// Input penalty nu
	static double nu;
		
	// Number of Iterations
	static int T;
		
	// Number of feature dimension
	static int d;
		
	// Number of data instances
	static int n;
		
	// Learning variables 
	static double w[];	
	
	// Bias
	static double b;
		
	// Probability Vector
	static double p[];
	
	// Id for Chosen feature
	static int jt;

	// Input eta value
	static double eta;
	
	// Data label vector
	static int vy[];
	
	// Bias vector for iterations
	static double vb[];
	
	// Average value of learning variables
	static double wavg[];
	
	// Define soft margin
	static int kexi[];
	
	// Sample in feature space
	static public int fSample() {
		int res=0;
		Random rnd=new Random();
		double r=rnd.nextDouble();
		double sum=0;
		for (int i=0;i<d;i++) {
			sum=sum+w[i]*w[i];
			if (r<sum) break;
			res=res+1;
		}
		return res;
	}
	
	// Parameter initialization
	static public void pInitial(String fname) throws Exception {
		// Need to set before-hand
		nu=1E-3;
		T=100;
		d=3231961;
		n=2376130;
		jt=0;
		eta=0.15;
		b=0;
		p=new double[n];
		w=new double[d];
		vb=new double[T];
		wavg=new double[d];
		kexi=new int[n];
		for (int i=0;i<n;i++)
			p[i]=1;
		double res=0;
		for (int i=0;i<n;i++)
			res=res+p[i]*p[i];
		res=Math.sqrt(res);
		for (int i=0;i<n;i++)
			p[i]=p[i]/res;
		for (int i=0;i<d;i++)
			w[i]=0;
		for (int i=0;i<d;i++)
			wavg[i]=0;
		
		// Load data label vector vy
		vy=new int[n];
		File fin=new File(fname);
		FileReader reader = new FileReader(fin);
		BufferedReader buf= new BufferedReader(reader);
		String line="";
		for (int i=0;i<n;i++) {
			line=buf.readLine();
			vy[i]=Integer.parseInt(line);
		}
		buf.close();
		reader.close();		
	}
	
	// Update p[] after dual step
	static public void pUpdate(int k) throws Exception {
		
		// Read data from hdfs file
		double tmp;
		int index;
		String uri = "sublinear/tmp/dual"+(new Integer(k)).toString()+"/part-r-00000";
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(URI.create(uri), conf);
		FSDataInputStream in = null;
		try {
			ByteArrayOutputStream baos = new ByteArrayOutputStream();   
			in = fs.open(new Path(uri));
			IOUtils.copyBytes(in, baos, 100000000, false); // 100M
			String str = baos.toString();
			str = str.substring(0, str.length() - 1);
			StringTokenizer itr = new StringTokenizer(str);
			for (int i=0;i<n;i++) {
				index=Integer.parseInt(itr.nextToken());
				tmp=Double.parseDouble(itr.nextToken());
				p[index] = p[index]*tmp;
			}
		} finally {
			IOUtils.closeStream(in);
		}
		
		// Normalization
		double res=0;
		for (int i=0;i<n;i++)
			res=res+p[i]*p[i];
		res=Math.sqrt(res);
		for (int i=0;i<n;i++)
			p[i]=p[i]/res;
	}
	
	// Update w[] after primal step
	static public void wUpdate(int k) throws Exception {

		// Read data from hdfs file
		double tmp;
		int index;
		Configuration conf = new Configuration();
		for (int i=0;i<30;i++) {
			String uri;
			if (i<10)
				uri = "sublinear/tmp/primal"+(new Integer(k)).toString()+"/part-r-0000"+(new Integer(i)).toString();
			else 
				uri = "sublinear/tmp/primal"+(new Integer(k)).toString()+"/part-r-000"+(new Integer(i)).toString();
			FileSystem fs = FileSystem.get(URI.create(uri), conf);
			FSDataInputStream in = null;
			try {
				ByteArrayOutputStream baos = new ByteArrayOutputStream();   
				in = fs.open(new Path(uri));
				IOUtils.copyBytes(in, baos, 10000000, false); // 10M
				String str = baos.toString();
				str = str.substring(0, str.length() - 1);
				StringTokenizer itr = new StringTokenizer(str);
				while (itr.hasMoreTokens()) {
					index=Integer.parseInt(itr.nextToken());
					tmp=Double.parseDouble(itr.nextToken());
					w[index-1] += tmp;
				}
			} finally {
				IOUtils.closeStream(in);
			}
		}
		
		// Normalization
		double res=0;
		for (int i=0;i<d;i++)
			res=res+w[i]*w[i];
		res=Math.sqrt(res);
		for (int i=0;i<d;i++)
			w[i]=w[i]/res;
		
		// Set wavg[]
		for (int i=0;i<d;i++)
			wavg[i]=wavg[i]*(k)/(k+1)+w[i]/(k+1);
		
		// Update kexi
		for (int i=0;i<n;i++)
			kexi[i]=0;
		for (int i=0;i<n;i++)
			if (p[i]>nu+1/Math.sqrt(n)) kexi[i]=2;
		
		// Update b
		res=0;
		for (int i=0;i<n;i++)
			res+=p[i]*vy[i];
		if (res>0) b=1;
		else {
			if (res<0) b=-1;
			else b=0;
		}
		vb[k]=b;
				
		// Sample a feature
		jt=fSample();
	}
	
	// Store w[] and p[] (if needed) in hdfs file 
	static public void pass_throgh_hdfs(int opt) throws Exception {
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		
		// Store w[] in "sublinear/tmp/paraw"
		Path outFile = new Path("sublinear/tmp/paraw");
		FSDataOutputStream out = fs.create(outFile);
		for (int i=0;i<d;i++) {
			out.writeDouble(w[i]);
			out.flush();
		}
		out.close();
		
		// Store p[] in "sublinear/tmp/parap"
		if (opt==0) {
			outFile = new Path("sublinear/tmp/parap");
			out = fs.create(outFile);
			for (int i=0;i<n;i++) {
				out.writeDouble(p[i]);
				out.flush();
			}
			out.close();
		}
		
		if (opt==1) {
			outFile = new Path("sublinear/tmp/parakexi");
			out = fs.create(outFile);
			for (int i=0;i<n;i++) {
				out.writeInt(kexi[i]);
				out.flush();
			}
			out.close();
		}
	}
	
	// Store parameters in different files according to iterations
	static public void logParameter(int k) throws IOException {
		File f=new File("output/Iteration-"+k+".log");
		FileWriter fw=new FileWriter(f);
		BufferedWriter bw=new BufferedWriter(fw);
		bw.write("Vector W:\n");
		for (int i=0;i<d;i++) {
			bw.write(w[i]+" ");
		}
		bw.write("\n");
		bw.write("b: "+b+"\n");
		bw.write("jt: "+jt+"\n");
		bw.write("Vector P:\n");
		for (int i=0;i<n;i++) {
			bw.write(p[i]+" ");
		}
		bw.write("\n");
		bw.flush();
		bw.close();
		fw.close();
	}
	
	// Store all learned parameters in "output.txt"
	static public void pOutput(int k) throws IOException {
		
		// Compute average value of b
		double bavg;
		double sum=0;
		for (int i=0;i<T;i++)
			sum+=vb[i];
		bavg=sum/T;
		
		// Write the local file
		File f=new File("output/output"+k+".txt");
		FileWriter fw=new FileWriter(f);
		BufferedWriter bw=new BufferedWriter(fw);
		for (int i=0;i<d;i++)
			bw.write(wavg[i]+"\n");
		bw.write(bavg+"\n");
		bw.flush();
		bw.close();
		fw.close();
	}
	
	// Program Entry
	public static void main(String[] args)throws Exception {
		
		// Parameter initialization
		pInitial(args[1]);
		
		// Iterations for T times
		for (int k=0;k<T;k++) {
			
			// Store w[] and p[] in hdfs file 
			pass_throgh_hdfs(0);
			
			// Get configuration of the first Map-Reduce job 
			Configuration conf_primal=new Configuration();
			
			// Set previous hdfs files in cache
			DistributedCache.addCacheFile(new URI("sublinear/tmp/paraw"), conf_primal);
			DistributedCache.addCacheFile(new URI("sublinear/tmp/parap"), conf_primal);
			
			// Set output path in hdfs for the first Map-Reduce job 
			Path tempDir1 = new Path("sublinear/tmp/primal"+(new Integer(k)).toString());
			
			// Pass parameters for the first Map-Reduce job through configuration
			conf_primal.setInt("T", T);
			conf_primal.setInt("d", d);
			conf_primal.setInt("n", n);
			conf_primal.setFloat("b", (float)b);
			
			// Set up for the first Map-Reduce job 
			Job job_primal=new Job(conf_primal, "SublinearPrimal");
			job_primal.setJarByClass(SublinearLogisticLtwo.class);
			FileInputFormat.addInputPath(job_primal,new Path(args[0]));
			FileOutputFormat.setOutputPath(job_primal,tempDir1);             
			job_primal.setMapperClass(PrimalMap.class);
			job_primal.setCombinerClass(PrimalReduce.class);
			job_primal.setReducerClass(PrimalReduce.class);
			job_primal.setNumReduceTasks(30);
			job_primal.setOutputKeyClass(Text.class);
			job_primal.setOutputValueClass(Text.class);
			
			// Parallel Block
			job_primal.waitForCompletion(true);
			
			// Update learning variables in primal step
			wUpdate(k);
			
			// Get configuration of the second Map-Reduce job
			Configuration conf_dual=new Configuration();
			
			// Set output path in hdfs for the second Map-Reduce job 
			Path tempDir2 = new Path("sublinear/tmp/dual"+(new Integer(k)).toString());
			
			// Pass parameters for the second Map-Reduce job through configuration 
			conf_dual.setInt("d", d);
			conf_dual.setInt("n", n);
			conf_dual.setInt("jt", jt);
			conf_dual.setFloat("b", (float)b);
			conf_dual.setFloat("eta", (float)eta);
			
			// Store w[] in hdfs file 
			pass_throgh_hdfs(1);
			
			// Set previous hdfs files in cache
			DistributedCache.addCacheFile(new URI("sublinear/tmp/paraw"), conf_dual);
			DistributedCache.addCacheFile(new URI("sublinear/tmp/parakexi"), conf_dual);
			
			// Set up for the second Map-Reduce job 
			Job job_dual=new Job(conf_dual, "SublinearDual");
			job_dual.setJarByClass(SublinearLogisticLtwo.class);
			FileInputFormat.addInputPath(job_dual,new Path(args[0]));
			FileOutputFormat.setOutputPath(job_dual,tempDir2);             
			job_dual.setMapperClass(DualMap.class);
			job_dual.setNumReduceTasks(1);
			job_dual.setOutputKeyClass(IntWritable.class);
			job_dual.setOutputValueClass(Text.class);
			
			// Parallel Block
			job_dual.waitForCompletion(true);
			
			// Update probability vector in dual step
			pUpdate(k);
			
			// Store parameters in a log file for this iteration
			logParameter(k);
			pOutput(k);
		}
		
		// Store final results
		//pOutput();
	}
	
}