// code by jph
package ch.ethz.idsc.subare.plot;

import java.io.File;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

import org.jfree.chart.ChartUtils;
import org.jfree.data.category.DefaultCategoryDataset;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.io.HomeDirectory;
import junit.framework.TestCase;

public class HistogramTest extends TestCase {
  public void testSimple() throws IOException {
    Map<String, Tensor> map = new LinkedHashMap<>();
    {
      map.put("a", Tensors.fromString("{1,2,3,2,3}"));
      map.put("b", Tensors.fromString("{2,3,4,5,6}"));
    }
    DefaultCategoryDataset defaultCategoryDataset = CategoryDatasets.create(map, RealScalar.of(2));
    Histogram histogram = new Histogram(defaultCategoryDataset);
    histogram.setPlotLabel("some");
    File file = HomeDirectory.Pictures(getClass().getSimpleName() + ".png");
    assertFalse(file.exists());
    ChartUtils.saveChartAsPNG(file, histogram.jFreeChart(), 1024, 768);
    assertTrue(file.isFile());
    file.delete();
    assertFalse(file.exists());
  }
}
