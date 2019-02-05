// code by fluric
package ch.ethz.idsc.subare.analysis;

import java.io.File;
import java.util.List;
import java.util.Map;

import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;

import ch.ethz.idsc.subare.plot.ListPlot;
import ch.ethz.idsc.subare.plot.XYSeriesList;
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
    XYSeriesList xySeriesList = StaticHelper.create(XYs, names);
    // return a new chart containing the overlaid plot...
    try {
      plot(path, path, "Number batches", "Error", xySeriesList, directory());
    } catch (Exception e) {
      System.err.println();
      e.printStackTrace();
    }
  }

  public static void createPlot(Map<String, Tensor> map, String path, List<DiscreteModelErrorAnalysis> errorAnalysisList) {
    for (int index = 0; index < errorAnalysisList.size(); ++index) {
      XYSeriesList xySeriesList = StaticHelper.create(map, index);
      // return a new chart containing the overlaid plot...
      String subPath = path + "_" + errorAnalysisList.get(index).name().toLowerCase();
      try {
        plot(subPath, subPath, "Number batches", "Error", xySeriesList, directory());
      } catch (Exception e) {
        System.err.println();
        e.printStackTrace();
      }
    }
  }

  public static void createPlot(Map<String, Tensor> map, String path) {
    XYSeriesList xySeriesList = StaticHelper.create(map);
    // return a new chart containing the overlaid plot...
    try {
      plot(path, path, "Number batches", "Error", xySeriesList, directory());
    } catch (Exception e) {
      System.err.println();
      e.printStackTrace();
    }
  }

  private static File plot( //
      String filename, String diagramTitle, String axisLabelX, String axisLabelY, XYSeriesList xySeriesList, File directory) throws Exception {
    JFreeChart jFreeChart = create(diagramTitle, axisLabelX, axisLabelY, xySeriesList);
    return savePlot(directory, filename, jFreeChart);
  }

  private static JFreeChart create(String diagramTitle, String axisLabelX, String axisLabelY, XYSeriesList xySeriesList) {
    ListPlot listPlot = new ListPlot(xySeriesList);
    listPlot.setPlotLabel(diagramTitle);
    listPlot.setAxisLabelX(axisLabelX);
    listPlot.setAxisLabelY(axisLabelY);
    return listPlot.jFreeChart();
  }

  private static File savePlot(File directory, String fileTitle, JFreeChart jFreeChart) throws Exception {
    File fileChart = new File(directory, fileTitle + ".png");
    ChartUtils.saveChartAsPNG(fileChart, jFreeChart, WIDTH, HEIGHT);
    GlobalAssert.that(fileChart.isFile());
    System.out.println("Exported " + fileTitle + ".png to " + directory);
    return fileChart;
  }
}
