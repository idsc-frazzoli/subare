// code by fluric
package ch.ethz.idsc.subare.analysis;

import java.io.File;
import java.util.List;
import java.util.Map;

import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;

import ch.ethz.idsc.subare.plot.ListPlot;
import ch.ethz.idsc.subare.plot.VisualSet;
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
    VisualSet visualSet = StaticHelper.create(XYs, names);
    // return a new chart containing the overlaid plot...
    try {
      plot(path, path, "Number batches", "Error", visualSet, directory());
    } catch (Exception e) {
      System.err.println();
      e.printStackTrace();
    }
  }

  public static void createPlot(Map<String, Tensor> map, String path, List<DiscreteModelErrorAnalysis> errorAnalysisList) {
    for (int index = 0; index < errorAnalysisList.size(); ++index) {
      VisualSet visualSet = StaticHelper.create(map, index);
      // return a new chart containing the overlaid plot...
      String subPath = path + "_" + errorAnalysisList.get(index).name().toLowerCase();
      try {
        plot(subPath, subPath, "Number batches", "Error", visualSet, directory());
      } catch (Exception e) {
        System.err.println();
        e.printStackTrace();
      }
    }
  }

  public static void createPlot(Map<String, Tensor> map, String path) {
    VisualSet visualSet = StaticHelper.create(map);
    try {
      plot(path, path, "Number batches", "Error", visualSet, directory());
    } catch (Exception e) {
      System.err.println();
      e.printStackTrace();
    }
  }

  private static File plot( //
      String filename, //
      String diagramTitle, //
      String axisLabelX, //
      String axisLabelY, //
      VisualSet visualSet, File directory) throws Exception {
    visualSet.setPlotLabel(diagramTitle);
    visualSet.setAxesLabelX(axisLabelX);
    visualSet.setAxesLabelY(axisLabelY);
    JFreeChart jFreeChart = ListPlot.of(visualSet);
    return savePlot(directory, filename, jFreeChart);
  }

  private static File savePlot(File directory, String fileTitle, JFreeChart jFreeChart) throws Exception {
    File file = new File(directory, fileTitle + ".png");
    ChartUtils.saveChartAsPNG(file, jFreeChart, WIDTH, HEIGHT);
    GlobalAssert.that(file.isFile());
    System.out.println("Exported " + fileTitle + ".png to " + directory);
    return file;
  }
}
