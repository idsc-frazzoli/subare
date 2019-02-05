// code by jph
package ch.ethz.idsc.subare.plot;

import org.jfree.data.category.DefaultCategoryDataset;

import ch.ethz.idsc.tensor.Tensor;

public class HistogramDataset {
  private final DefaultCategoryDataset defaultCategoryDataset = new DefaultCategoryDataset();
  private final LabelList labelRows = new LabelList();
  private final LabelList labelCols = new LabelList();

  public HistogramRow add(Tensor vector) {
    HistogramRow histogramRow = new HistogramRow(labelRows.next());
    for (int index = 0; index < vector.length(); ++index)
      defaultCategoryDataset.addValue(vector.Get(index).number(), histogramRow.comparableLabel(), labelCols.get(index));
    return histogramRow;
  }

  public void setColumnNames(Object[] objects) {
    for (int index = 0; index < objects.length; ++index)
      labelCols.get(index).setString(objects[index].toString());
  }

  public DefaultCategoryDataset defaultCategoryDataset() {
    return defaultCategoryDataset;
  }
}
