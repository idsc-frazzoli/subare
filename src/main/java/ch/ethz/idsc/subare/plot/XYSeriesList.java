// code by jph
package ch.ethz.idsc.subare.plot;

import java.util.stream.IntStream;

import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import ch.ethz.idsc.tensor.Tensor;

public class XYSeriesList {
  private final XYSeriesCollection xySeriesCollection = new XYSeriesCollection();

  /** Mathematica::ListPlot[points]
   * 
   * @param points of the form {{x1, y1}, {x2, y2}, ..., {xn, yn}}
   * @return */
  public XYSeries add(Tensor points) {
    XYSeries xySeries = new XYSeries(xySeriesCollection.getSeriesCount());
    for (Tensor row : points)
      xySeries.add(row.Get(0).number(), row.Get(1).number());
    xySeriesCollection.addSeries(xySeries);
    return xySeries;
  }

  /** MATLAB::plot(x, y)
   * 
   * @param domain {x1, x2, ..., xn}
   * @param values {y1, y2, ..., yn}
   * @return */
  public XYSeries addSeries(Tensor domain, Tensor values) {
    XYSeries xySeries = new XYSeries(xySeriesCollection.getSeriesCount());
    IntStream.range(0, domain.length()) //
        .forEach(index -> xySeries.add(domain.Get(index).number(), values.Get(index).number()));
    xySeriesCollection.addSeries(xySeries);
    return xySeries;
  }

  public XYSeriesCollection xySeriesCollection() {
    return xySeriesCollection;
  }
}
