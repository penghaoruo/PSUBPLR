import java.io.*;

public class DataProcess {
	public static void generateTrain() throws Exception {
		int index=0;
		for (int i=0;i<120;i++) {
			String fname="Day"+i+".svm";
			File fin=new File("url_svmlight/"+fname);
			FileReader reader = new FileReader(fin);
			BufferedReader buf= new BufferedReader(reader);
			File fout=new File("train/"+fname);
			FileWriter fw=new FileWriter(fout);
			BufferedWriter bw=new BufferedWriter(fw);
			
			String line=buf.readLine();
			while (line!=null) {
				index++;
				bw.write(index+" ");
				bw.flush();
				int y=0;
				if (line.charAt(0)=='-') y=-1;
				if (line.charAt(0)=='+') y=1;
				line=line.substring(3);
				bw.write(line+" ");
				bw.flush();
				String[] strs=line.split(" ");
				double sum=0;
				for (int j=0;j<strs.length;j++) {
					double tmp=Double.parseDouble(strs[j].split(":")[1]);
					sum=sum+tmp*tmp;
				}
				sum=Math.sqrt(sum);
				bw.write(sum+" "+y+"\r\n");
				bw.flush();
				line=buf.readLine();
			}
			System.out.println(i);
			buf.close();
			reader.close();
			bw.close();
			fw.close();
		}
	}
	public static void test_single() throws Exception {
		int d=3231961;
		int n=20000;
		double w[]=new double[d];
		double res[]=new double[n];
		double b=0;
		int y[]=new int[n];
		File fin=new File("output.txt");
		FileReader reader = new FileReader(fin);
		BufferedReader buf= new BufferedReader(reader);
		String line="";
		for (int i=0;i<d;i++) {
			line=buf.readLine();
			if (line.equals("")) line=buf.readLine();
			w[i]=Double.parseDouble(line);
		}
		line=buf.readLine();
		if (line.equals("")) line=buf.readLine();
		b=Double.parseDouble(line);
		buf.close();
		reader.close();
		
		fin=new File("test.svm");
		reader = new FileReader(fin);
		buf= new BufferedReader(reader);
		for (int i=0;i<n;i++) {
			line=buf.readLine();
			double x[]=new double[d];
			if (line.charAt(0)=='-') y[i]=-1;
			if (line.charAt(0)=='+') y[i]=1;
			line=line.substring(3);
			String[] strs=line.split(" ");
			for (int j=0;j<d;j++) x[j]=0;
			for (int j=0;j<strs.length;j++)
				x[Integer.parseInt(strs[j].split(":")[0])]=Double.parseDouble(strs[j].split(":")[1]);
			double sum=0;
			for (int j=0;j<d;j++)
				sum+=w[j]*x[j];
			sum+=b;
			res[i]=sum;
			System.out.println(i);
		}
		buf.close();
		reader.close();
		
		File fout=new File("performance.txt");
		FileWriter fw=new FileWriter(fout);
		BufferedWriter bw=new BufferedWriter(fw);
		int k=0;
		for (int i=0;i<n;i++)
			if (res[i]*y[i]>0) k++;
		bw.write("Accuracy: "+(double)k/(double)n+"\r\n");
		for (int i=0;i<n;i++)
			bw.write(res[i]+" "+y[i]+"\r\n");
		bw.close();
		fw.close();
	}
	public static void generateY() throws Exception {
		File fout=new File("ylabel");
		FileWriter fw=new FileWriter(fout);
		BufferedWriter bw=new BufferedWriter(fw);
		for (int i=0;i<120;i++) {
			String fname="Day"+i+".svm";
			File fin=new File("url_svmlight/"+fname);
			FileReader reader = new FileReader(fin);
			BufferedReader buf= new BufferedReader(reader);

			String line=buf.readLine();
			while (line!=null) {
				int y=0;
				if (line.charAt(0)=='-') y=-1;
				if (line.charAt(0)=='+') y=1;
				line=line.substring(3);
				bw.write(y+"\n");
				bw.flush();
				line=buf.readLine();
			}
			System.out.println(i);
			buf.close();
			reader.close();
		}
		bw.close();
		fw.close();
	}
	public static void test_all() throws Exception {
		int d=3231961;
		int n=20000;
		double w[]=new double[d];
		double b=0;
		int y=0;
		
		File fout=new File("performance.txt");
		FileWriter fw=new FileWriter(fout);
		BufferedWriter bw=new BufferedWriter(fw);
		
		for (int k=0;k<10;k++) {
			int num=0;
			
			File fin=new File("7/output"+(k*10)+".txt");
			FileReader reader = new FileReader(fin);
			BufferedReader buf= new BufferedReader(reader);
			String line="";
			for (int i=0;i<d;i++) {
				line=buf.readLine();
				if (line.equals("")) line=buf.readLine();
				w[i]=Double.parseDouble(line);
			}
			line=buf.readLine();
			if (line.equals("")) line=buf.readLine();
			b=Double.parseDouble(line);
			buf.close();
			reader.close();
			
			fin=new File("test.svm");
			reader = new FileReader(fin);
			buf= new BufferedReader(reader);
			for (int i=0;i<n;i++) {
				line=buf.readLine();
				double x[]=new double[d];
				if (line.charAt(0)=='-') y=-1;
				if (line.charAt(0)=='+') y=1;
				line=line.substring(3);
				String[] strs=line.split(" ");
				for (int j=0;j<d;j++) x[j]=0;
				for (int j=0;j<strs.length;j++)
					x[Integer.parseInt(strs[j].split(":")[0])]=Double.parseDouble(strs[j].split(":")[1]);
				double sum=0;
				for (int j=0;j<d;j++)
					sum+=w[j]*x[j];
				sum+=b;
				if (sum*y>0) num++; 
				
			}
			buf.close();
			reader.close();
			bw.write((double)num/(double)n+"\r\n");
			bw.flush();
			System.out.println(k);
		}
		bw.close();
		fw.close();
	}
	public static void main(String[] args) throws Exception {
		//generateTrain();
		//generateY();
		test_single();
		//test_all();
	}
}