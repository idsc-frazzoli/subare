// code by jph
package ch.ethz.idsc.subare.ch02;

import ch.ethz.idsc.subare.util.FairArg;
import ch.ethz.idsc.tensor.Tensor;

public abstract class FairArgAgent extends Agent {
  protected abstract Tensor getQVector();

  @Override
  public final int protected_takeAction() {
    FairArg fairArgMax = FairArg.max(getQVector());
    if (!fairArgMax.isUnique()) {
      if (getCount().number().intValue() < openingSequence.size())
        return openingSequence.get(getCount().number().intValue());
      notifyAboutRandomizedDecision();
    }
    return fairArgMax.nextRandomIndex(); // (2.2)
  }
}
