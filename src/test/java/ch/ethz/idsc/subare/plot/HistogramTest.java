// code by jph
package ch.ethz.idsc.subare.plot;

import java.io.File;
import java.io.IOException;

import org.jfree.chart.ChartUtils;

import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.io.HomeDirectory;
import junit.framework.TestCase;

public class HistogramTest extends TestCase {
  public void testSimple() throws IOException {
    HistogramDataset histogramDataset = new HistogramDataset();
    HistogramRow histogramRow = histogramDataset.add(Tensors.vector(1, 2, 3, 2, 3));
    // histogramRow.setChartLegend("first");
    histogramDataset.add(Tensors.vector(2, 3, 4, 5, 6));
    histogramDataset.add(Tensors.vector(2, 3, 4, 5, 6, 2, 3, 4, 5));
    // histogramDataset.setRowNames(new String[] { "p1", "short", "longer" });
    histogramDataset.setColumnNames(new String[] { "aa1", "bb1", "cc1" });
    histogramDataset.setColumnNames(Tensors.vector(1, 2, 3).stream().toArray());
    Histogram histogram = new Histogram(histogramDataset.defaultCategoryDataset());
    histogram.setPlotLabel("some");
    File file = HomeDirectory.Pictures(getClass().getSimpleName() + ".png");
    // assertFalse(file.exists());
    ChartUtils.saveChartAsPNG(file, histogram.jFreeChart(), 1024, 768);
    // assertTrue(file.isFile());
    // file.delete();
    // assertFalse(file.exists());
  }
}
