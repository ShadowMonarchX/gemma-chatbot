import { useEffect, useMemo, useRef, type FC } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';

import type { UiMessage } from '../api/types';
import MessageBubble from './MessageBubble';
import TypingIndicator from './TypingIndicator';

export interface MessageListProps {
  messages: UiMessage[];
  isStreaming: boolean;
}

const MessageList: FC<MessageListProps> = ({ messages, isStreaming }) => {
  const listRef = useRef<HTMLDivElement | null>(null);

  const shouldVirtualize = messages.length > 50;

  const virtualizer = useVirtualizer({
    count: shouldVirtualize ? messages.length : 0,
    getScrollElement: () => listRef.current,
    estimateSize: () => 120,
    overscan: 8,
  });

  const virtualItems = virtualizer.getVirtualItems();

  useEffect(() => {
    if (!listRef.current) {
      return;
    }

    if (shouldVirtualize) {
      virtualizer.scrollToIndex(messages.length - 1, { align: 'end' });
      return;
    }

    listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [messages, shouldVirtualize, virtualizer]);

  const content = useMemo(() => {
    if (!shouldVirtualize) {
      return messages.map((message) => (
        <MessageBubble
          key={message.id}
          message={message}
        />
      ));
    }

    return (
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          width: '100%',
          position: 'relative',
        }}
      >
        {virtualItems.map((virtualRow) => {
          const message = messages[virtualRow.index];
          return (
            <div
              key={message.id}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                transform: `translateY(${virtualRow.start}px)`,
              }}
            >
              <MessageBubble message={message} />
            </div>
          );
        })}
      </div>
    );
  }, [messages, shouldVirtualize, virtualItems, virtualizer]);

  return (
    <div
      ref={listRef}
      className="flex h-full flex-col gap-4 overflow-y-auto px-1 pb-4"
    >
      {content}
      <TypingIndicator visible={isStreaming} />
    </div>
  );
};

export default MessageList;
