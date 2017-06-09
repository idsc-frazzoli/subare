// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.StepInterface;

/** class steps through a given episode and stores the steps for one or multiple replays */
public class EpisodeRecording {
  private final List<StepInterface> list = new LinkedList<>();

  public EpisodeRecording(EpisodeInterface episodeInterface) {
    while (episodeInterface.hasNext()) {
      StepInterface stepInterface = episodeInterface.step();
      list.add(stepInterface);
    }
  }

  public EpisodeInterface replay() {
    return new EpisodeInterface() {
      final Iterator<StepInterface> iterator = list.iterator();

      @Override
      public StepInterface step() {
        return iterator.next();
      }

      @Override
      public boolean hasNext() {
        return iterator.hasNext();
      }
    };
  }
}
