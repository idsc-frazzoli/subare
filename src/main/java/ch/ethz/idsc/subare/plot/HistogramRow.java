// code by jph
package ch.ethz.idsc.subare.plot;

public class HistogramRow {
  private final ComparableLabel comparableLabel;

  public HistogramRow(ComparableLabel comparableLabel) {
    this.comparableLabel = comparableLabel;
  }

  public void setChartLegend(String string) {
    comparableLabel.setString(string);
  }

  ComparableLabel comparableLabel() {
    return comparableLabel;
  }
}
