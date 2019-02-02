// code by fluric
package ch.ethz.idsc.subare.analysis;

import java.io.File;
import java.util.List;
import java.util.Map;

import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYDataset;

import ch.ethz.idsc.subare.plot.ListPlotBuilder;
import ch.ethz.idsc.subare.plot.XYDatasets;
import ch.ethz.idsc.subare.util.GlobalAssert;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.io.HomeDirectory;

public enum BatchesPlotUtils {
  ;
  private static final int WIDTH = 1280;
  private static final int HEIGHT = 720;

  private static File directory() {
    File directory = HomeDirectory.Pictures("plots");
    directory.mkdir();
    return directory;
  }

  /** A plot can be created and is stored at {@code plot/path}. XYs contains a list of tensors (data sets) that again
   * contains tensors with x and y components. The list of names contains the data set names.
   * @param XYs
   * @param names */
  public static void createPlot(List<Tensor> XYs, List<String> names, String path) {
    XYDataset xyDataset = XYDatasets.create(XYs, names);
    // return a new chart containing the overlaid plot...
    try {
      plot(path, path, "Number batches", "Error", xyDataset, directory());
    } catch (Exception e) {
      System.err.println();
      e.printStackTrace();
    }
  }

  public static void createPlot(Map<String, Tensor> map, String path, List<DiscreteModelErrorAnalysis> errorAnalysisList) {
    for (int index = 0; index < errorAnalysisList.size(); ++index) {
      XYDataset xyDataset = XYDatasets.create(map, index);
      // return a new chart containing the overlaid plot...
      String subPath = path + "_" + errorAnalysisList.get(index).name().toLowerCase();
      try {
        plot(subPath, subPath, "Number batches", "Error", xyDataset, directory());
      } catch (Exception e) {
        System.err.println();
        e.printStackTrace();
      }
    }
  }

  public static void createPlot(Map<String, Tensor> map, String path) {
    XYDataset xyDataset = XYDatasets.create(map);
    // return a new chart containing the overlaid plot...
    try {
      plot(path, path, "Number batches", "Error", xyDataset, directory());
    } catch (Exception e) {
      System.err.println();
      e.printStackTrace();
    }
  }

  private static File plot( //
      String filename, String diagramTitle, String axisLabelX, String axisLabelY, XYDataset dataset, File directory) throws Exception {
    JFreeChart jFreeChart = create(diagramTitle, axisLabelX, axisLabelY, dataset);
    return savePlot(directory, filename, jFreeChart);
  }

  private static JFreeChart create(String diagramTitle, String axisLabelX, String axisLabelY, XYDataset dataset) {
    return new ListPlotBuilder(diagramTitle, axisLabelX, axisLabelY, dataset).getJFreeChart();
  }

  private static File savePlot(File directory, String fileTitle, JFreeChart jFreeChart) throws Exception {
    File fileChart = new File(directory, fileTitle + ".png");
    ChartUtils.saveChartAsPNG(fileChart, jFreeChart, WIDTH, HEIGHT);
    GlobalAssert.that(fileChart.isFile());
    System.out.println("Exported " + fileTitle + ".png to " + directory);
    return fileChart;
  }
}
