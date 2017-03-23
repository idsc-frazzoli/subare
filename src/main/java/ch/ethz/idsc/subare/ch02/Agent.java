// code by jph
package ch.ethz.idsc.subare.ch02;

import java.util.Random;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.ZeroScalar;

public abstract class Agent {
  protected static final Random random = new Random();
  // ---
  private Scalar total = ZeroScalar.get();
  private int count = 0;
  private Tensor actions = Tensors.empty();
  private Tensor qvalues = Tensors.empty();

  public abstract int takeAction();

  protected abstract Tensor protected_QValues();

  protected abstract void protected_feedback(int a, Scalar value);

  public final void feedback(int a, Scalar value) {
    ++count;
    total = total.add(value);
    actions.append(RealScalar.of(a));
    qvalues.append(protected_QValues());
    protected_feedback(a, value);
  }

  public final Scalar getCount() {
    return RealScalar.of(count);
  }

  public Scalar getTotal() {
    return total;
  }

  public Scalar getAverage() {
    return total.divide(getCount());
  }

  public Tensor getActions() {
    return actions.unmodifiable();
  }

  public Tensor getQValues() {
    return qvalues.unmodifiable();
  }

  public abstract String getDescription();

  public final String getAbsDesc() {
    String name = getClass().getSimpleName();
    name = name.substring(0, name.length() - 5); // drop "Agent"
    return String.format("%10s%15s", name, getDescription());
  }

  @Override
  public String toString() {
    return getAbsDesc();
  }
}
